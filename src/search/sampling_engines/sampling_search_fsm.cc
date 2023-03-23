#include "sampling_search_fsm.h"

#include "sampling_search_base.h"
#include "sampling_engine.h"

#include "../option_parser.h"
#include "../plugin.h"

#include "../task_utils/task_properties.h"
#include "../task_utils/successor_generator.h"

#include "../utils/timer.h"

#include <sstream>
#include <fstream>
#include <string>
#include <chrono>
#include <queue>
#include <iterator>
#include <climits>

using namespace std;

namespace sampling_engine {

string SamplingSearchFsm::construct_header() const {
    ostringstream oss;

    oss << "#<Time>=" << utils::g_timer << "/" << sampling_technique::max_time << endl;
    if (store_plan_cost){
        oss << "#<PlanCost>=single integer value" << endl;
    }
    if (store_state) {
        oss << "#<State>=";
        for (unsigned i = 0; i < relevant_facts.size(); i++)
            oss << task->get_fact_name(relevant_facts[i]) << state_separator;
        oss.seekp(-1,oss.cur);
    }

    return oss.str();
}

string SamplingSearchFsm::sample_file_header() const {
    return header;
}

vector<string> SamplingSearchFsm::format_output(vector<shared_ptr<PartialAssignment>>& samples) {
    utils::g_log << "[Sampling Engine] Formatting the output..." << endl;

    vector<string> lines;
    for (shared_ptr<PartialAssignment>& s: samples) {
        ostringstream line;
        if (store_plan_cost)
            line << s->estimated_heuristic << field_separator;
        if (store_state)
            line << s->to_binary();
        lines.push_back(line.str());
    }
    return lines;
}

void SamplingSearchFsm::successor_improvement() {
    if (sai_partial) {
        utils::g_log << "[SAI] SAI in partial states will be done implicitly with the SUI." << endl;
        for (pair<const size_t,sampling_technique::SuiNode> p : sampling_technique::sui_mapping) {
            for (shared_ptr<PartialAssignment>& s : p.second.samples)
                s->estimated_heuristic = p.second.best_h;
        }
    }

    // SUI loop
    utils::g_log << "[SUI] Updating the h-values..." << endl;
    auto t = std::chrono::high_resolution_clock::now();
    bool relaxed, any_relaxed;
    do {
        any_relaxed = false;
        for (pair<const size_t,sampling_technique::SuiNode> s: sampling_technique::sui_mapping) {
            relaxed = false;
            for (pair<size_t,int> s_ : s.second.successors) { // pair<state,op_cost>
                assert(sampling_technique::sui_mapping.find(s_.first) != sampling_technique::sui_mapping.end());
                int candidate_heuristic = sampling_technique::sui_mapping[s_.first].best_h + (unit_cost ? 1 : s_.second);
                if (candidate_heuristic < sampling_technique::sui_mapping[s.first].best_h) {
                    sampling_technique::sui_mapping[s.first].best_h = candidate_heuristic;
                    relaxed = true;
                    for (shared_ptr<PartialAssignment>& s : s.second.samples)
                        s->estimated_heuristic = candidate_heuristic;
                }
            }
            if (relaxed)
                any_relaxed = true;
        }
    } while (any_relaxed);

    utils::g_log << "[SUI] Time updating h-values: " << fixed << (std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t).count() / 1000.0) << "s." << endl;
}

void SamplingSearchFsm::sample_improvement(vector<shared_ptr<PartialAssignment>>& samples) {
    auto t_sai = std::chrono::high_resolution_clock::now();
    utils::g_log << "[SAI] Computing SAI..." << endl;

    // Mapping where each state will have a pair, where the first element is the
    // smallest h-value found for the state and the second is a list of pointers
    // to all h-values vars of all identical states.
    unordered_map<string,pair<int,vector<int*>>> pairs;

    for (shared_ptr<PartialAssignment>& s: samples) {
        string bin = s->to_binary(true);
        if (pairs.count(bin) == 0) {
            pairs[bin] = make_pair(
                s->estimated_heuristic,
                vector<int*>{&s->estimated_heuristic}
            );
        } else {
            pairs[bin].first = min(pairs[bin].first, s->estimated_heuristic);
            pairs[bin].second.push_back(&s->estimated_heuristic);
        }
    }
    int updates = 0;
    for (pair<string,pair<int,vector<int*>>> p : pairs) {
        for (int* h_ptr : p.second.second) {
            if (*h_ptr != p.second.first) {
                assert(*h_ptr > p.second.first);
                *h_ptr = p.second.first;
                updates++;
            }
        }
    }
    utils::g_log << "[SAI] Updated samples: " << updates << endl;
    utils::g_log << "[SAI] Done in " << fixed << (std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_sai).count() / 1000.0) << "s." << endl;
}

bool is_number(const string& s) {
    return !s.empty() && find_if(s.begin(), s.end(), [](unsigned char c) { return !isdigit(c); }) == s.end();
}

vector<string> SamplingSearchFsm::extract_samples() {
    utils::g_log << "[Sampling Engine] Extracting samples..." << endl;
    utils::g_log << "[Sampling Engine] " << sampling_technique::modified_tasks.size() << " samples obtained in sampling." << endl;

    if (sui)
        successor_improvement();
    assert(!sai_partial || (sai_partial && sui));

    // Random samples 1st step: Add random samples to the sample set
    bool has_random_samples = !sampling_technique::random_modified_tasks.empty();
    int random_sample_h;
    if (has_random_samples) {
        random_sample_h = sampling_technique::random_modified_tasks[0]->estimated_heuristic;
        // add random_modified_tasks to modified_tasks
        int final_num_random_samples =
                (sampling_technique::modified_tasks.size() * sampling_technique::random_samples_percentage) / (1.0 - sampling_technique::random_samples_percentage);
        utils::g_log << "[Random Samples] Adding " << final_num_random_samples << " random samples to the sample set." << endl;
        for (int i = 0; i < final_num_random_samples; i++)
            sampling_technique::modified_tasks.push_back(sampling_technique::random_modified_tasks[i]);
        sampling_technique::random_modified_tasks.clear();
    }

    if (sai_complete)
        sample_improvement(sampling_technique::modified_tasks);

    // Random samples 2nd step: Update h-value of random samples to smallest h-value seen during sampling (after SAI and/or SUI)
    if (has_random_samples) {
        int max_regression_h = -1;
        for (shared_ptr<PartialAssignment>& pa : sampling_technique::modified_tasks) {
            if (pa->estimated_heuristic == random_sample_h)
                break;
            max_regression_h = max(max_regression_h, pa->estimated_heuristic);
        }
        for (shared_ptr<PartialAssignment>& pa : sampling_technique::modified_tasks) {
            if (pa->estimated_heuristic == random_sample_h)
                pa->estimated_heuristic = max_regression_h + 1;
        }
    }

    header = construct_header();
    return format_output(sampling_technique::modified_tasks);
}

SamplingSearchFsm::SamplingSearchFsm(const options::Options &opts)
    : SamplingSearchBase(opts),
      store_plan_cost(opts.get<bool>("store_plan_cost")),
      store_state(opts.get<bool>("store_state")),
      sai_partial(opts.get<string>("sai") == "partial" || opts.get<string>("sai") == "both"),
      sai_complete(opts.get<string>("sai") == "complete" || opts.get<string>("sai") == "both"),
      random_multiplier(opts.get<int>("random_multiplier")),
      relevant_facts(task_properties::get_strips_fact_pairs(task.get())),
      registry(task_proxy),
      rng(utils::parse_rng_from_options(opts)) {

    sui = opts.get<bool>("sui");
}

static shared_ptr<SearchEngine> _parse_sampling_search_fsm(OptionParser &parser) {
    parser.document_synopsis("Sampling Search Manager", "");

    sampling_engine::SamplingSearchBase::add_sampling_search_base_options(parser);
    sampling_engine::SamplingEngine::add_sampling_options(parser);
    sampling_engine::SamplingStateEngine::add_sampling_state_options(
            parser, "fields", "pddl", ";", ";");

    parser.add_option<bool>(
            "store_plan_cost",
            "Store for every state its cost along the plan to the goal",
            "true");
    parser.add_option<bool>(
            "store_state",
            "Store every state along the plan",
            "true");
    parser.add_option<string>(
            "sai",
            "Identical states receive the best heuristic value assigned between them (SAI in: none, partial, complete, both).",
            "none");
    parser.add_option<bool>(
            "sui",
            "Correct h-values using SUI via K-step forward repeatedly",
            "false");
    parser.add_option<int>(
            "random_multiplier",
            "Value to multiply each random sample h-value.",
            "1");

    SearchEngine::add_options_to_parser(parser);
    Options opts = parser.parse();
    shared_ptr<sampling_engine::SamplingSearchFsm> engine;
    if (!parser.dry_run()) {
        engine = make_shared<sampling_engine::SamplingSearchFsm>(opts);
    }

    return engine;
}

static Plugin<SearchEngine> _plugin_search("sampling_search_fsm", _parse_sampling_search_fsm);

}
