#include <stack>
#include <algorithm>
#include <chrono>
#include <fstream>

#include "technique_gbackward_fsm.h"

#include "../evaluation_result.h"
#include "../heuristic.h"
#include "../plugin.h"

#include "../tasks/modified_init_goals_task.h"
#include "../tasks/partial_state_wrapper_task.h"

#include "../task_utils/sampling.h"

#include "../task_utils/task_properties.h"

#include "../sampling_engines/sampling_engine.h"

using namespace std;

namespace sampling_technique {
const string TechniqueGBackwardFsm::name = "gbackward_fsm";

static int compute_heuristic(
        const TaskProxy &task_proxy, Heuristic *bias,
        utils::RandomNumberGenerator &rng, const PartialAssignment &assignment) {
    auto pair_success_state = assignment.get_full_state(
            true, rng);
    if (!pair_success_state.first) {
        return EvaluationResult::INFTY;
    }
    StateRegistry state_registry(task_proxy);
    vector<int> initial_facts= pair_success_state.second.get_values();
    State state = state_registry.insert_state(move(initial_facts));

    return bias->compute_heuristic(state);
}

bool is_number(const string& s) {
    return !s.empty() && find_if(s.begin(),
        s.end(), [](unsigned char c) { return !isdigit(c); }) == s.end();
}

const string &TechniqueGBackwardFsm::get_name() const {
    return name;
}

TechniqueGBackwardFsm::TechniqueGBackwardFsm(const options::Options &opts)
        : SamplingTechnique(opts),
          technique(opts.get<string>("technique")),
          regression_depth(opts.get<string>("regression_depth")),
          depth_k(opts.get<int>("depth_k")),
          allow_duplicates_interrollout(
              opts.get<string>("allow_duplicates") == "all" || opts.get<string>("allow_duplicates") == "interrollout"
          ),
          allow_duplicates_intrarollout(
              opts.get<string>("allow_duplicates") == "all"
          ),
          wrap_partial_assignment(opts.get<bool>("wrap_partial_assignment")),
          deprioritize_undoing_steps(opts.get<bool>("deprioritize_undoing_steps")),
          is_valid_walk(opts.get<bool>("is_valid_walk")),
          restart_h_when_goal_state(opts.get<bool>("restart_h_when_goal_state")),
          state_filtering(opts.get<string>("state_filtering")),
          bfs_percentage(opts.get<double>("bfs_percentage")),
          bias_evaluator_tree(opts.get_parse_tree("bias", options::ParseTree())),
          bias_probabilistic(opts.get<bool>("bias_probabilistic")),
          bias_adapt(opts.get<double>("bias_adapt")),
          bias_reload_frequency(opts.get<int>("bias_reload_frequency")),
          bias_reload_counter(0),
          facts_file(opts.get<string>("facts_file")) {
    assert(technique == "rw" || technique == "bfs" || technique == "dfs" || technique == "bfs_rw");
    if (technique == "bfs_rw")
        assert(bfs_percentage >= 0.0 && bfs_percentage <= 1.0);
}

vector<shared_ptr<PartialAssignment>> TechniqueGBackwardFsm::sample_with_random_walk(
    PartialAssignment initial_state,
    const unsigned steps,
    const ValidStateDetector &is_valid_state,
    const PartialAssignmentBias *bias,
    const TaskProxy &task_proxy,
    const bool sample_initial_state,
    const bool global_hash_table,
    const utils::HashSet<PartialAssignment> states_to_avoid
) {
    PartialAssignment pa = initial_state;
    vector<shared_ptr<PartialAssignment>> samples;
    if (sample_initial_state) {
        shared_ptr<PartialAssignment> pa_ptr = make_shared<PartialAssignment>(pa);
        prepare_posprocessing(pa_ptr, task_proxy);
        samples.push_back(pa_ptr);
    }
    utils::HashSet<PartialAssignment> local_hash_table;
    utils::HashSet<PartialAssignment> *ht_pointer = global_hash_table ? &hash_table : &local_hash_table;
    bool renegerate_applicable_ops = true;
    while (samples.size() < steps && !stopped) {
        OperatorID applied_op = OperatorID::no_operator;
        PartialAssignment pa_ = rrws->sample_state_length(
            pa,
            1,
            applied_op,
            renegerate_applicable_ops,
            deprioritize_undoing_steps,
            is_valid_state,
            bias,
            bias_probabilistic,
            bias_adapt
        );
        assert(
            (pa_ == pa && applied_op == OperatorID::no_operator) ||
            (pa_ != pa && applied_op != OperatorID::no_operator)
        );
        if (pa_ == pa) // there is no applicable operator
            break;

        if ((allow_duplicates_intrarollout || ht_pointer->find(pa_) == ht_pointer->end())
                && (states_to_avoid.find(pa_) == states_to_avoid.end())) {
            if (restart_h_when_goal_state && task_properties::is_goal_assignment(task_proxy, pa_)) {
                pa_.estimated_heuristic = 0;
                pa_.states_to_goal = 0;
            } else {
                pa_.estimated_heuristic = pa.estimated_heuristic + (unit_cost ? 1 : operators[applied_op].get_cost());
                pa_.states_to_goal = pa.states_to_goal + 1;
            }

            vector<int> pa_values = pa_.get_values();
            PartialAssignment ps(pa_, move(pa_values));
            ps.estimated_heuristic = pa_.estimated_heuristic;
            ps.states_to_goal = pa_.states_to_goal;
            shared_ptr<PartialAssignment> ps_ = make_shared<PartialAssignment>(ps);
            if (!prepare_posprocessing(ps_, task_proxy))
                continue;
            samples.push_back(ps_);
            ht_pointer->insert(pa_);

            pa = pa_;
            renegerate_applicable_ops = true;
        } else {
            renegerate_applicable_ops = false;
        }
        stopped = stop_sampling();
    }

    assert(samples.size() <= steps);
    return samples;
}

vector<shared_ptr<PartialAssignment>> TechniqueGBackwardFsm::sample_with_bfs_or_dfs(
    string technique,
    PartialAssignment initial_state,
    const unsigned steps,
    const ValidStateDetector &is_valid_state,
    const TaskProxy &task_proxy
) {
    PartialAssignment pa = initial_state;
    vector<shared_ptr<PartialAssignment>> samples;
    stack<PartialAssignment> stack;
    queue<PartialAssignment> queue;

    if (technique == "dfs")
        stack.push(pa);
    else
        queue.push(pa);

    hash_table.insert(pa);
    while (samples.size() < steps && !stopped) {
        if (technique == "dfs") {
            if (stack.empty())
                break;
            pa = stack.top();
            stack.pop();
        } else {
            if (queue.empty())
                break;
            pa = queue.front();
            queue.pop();
        }

        shared_ptr<PartialAssignment> ps_ = make_shared<PartialAssignment>(pa);
        if (!prepare_posprocessing(ps_, task_proxy, technique == "bfs"))
            continue;
        samples.push_back(ps_);

        int idx_op = 0, rng_seed = (*rng)() * (INT32_MAX - 1);
        while (idx_op != -1 && !stopped) {
            OperatorID applied_op = OperatorID::no_operator;
            PartialAssignment pa_ = dfss->sample_state_length(
                pa,
                rng_seed,
                idx_op,
                applied_op,
                is_valid_state
            );
            // idx_op has the index of the operator that was used,
            // or -1 if all operators have already been checked
            assert(
                (idx_op == -1 && applied_op == OperatorID::no_operator) ||
                (idx_op != -1 && applied_op != OperatorID::no_operator)
            );
            if (idx_op == -1) {
                assert(pa == pa_);
                break;
            }
            if (pa_ == pa)
                continue;

            if (allow_duplicates_intrarollout || hash_table.find(pa_) == hash_table.end()) {
                if (restart_h_when_goal_state && task_properties::is_goal_assignment(task_proxy, pa_)) {
		            pa_.estimated_heuristic = 0;
                    pa_.states_to_goal = 0;
                } else {
                    pa_.estimated_heuristic = pa.estimated_heuristic + (unit_cost ? 1 : operators[applied_op].get_cost());
                    pa_.states_to_goal = pa.states_to_goal + 1;
                }

                if (pa_.states_to_goal <= depth_k) {
                    hash_table.insert(pa_);
                    if (technique == "dfs")
                        stack.push(pa_);
                    else
                        queue.push(pa_);
                }
            }
            idx_op++;
            stopped = stop_sampling();
        }
    }
    assert(samples.size() <= steps);
    return samples;
}

vector<shared_ptr<PartialAssignment>> TechniqueGBackwardFsm::sample_with_percentage_limited_bfs(
    double bfs_percentage,
    PartialAssignment initial_state,
    const ValidStateDetector &is_valid_state,
    vector<PartialAssignment> &leaves,
    const TaskProxy &task_proxy
) {
    assert(bfs_percentage >= 0.0 && bfs_percentage <= 1.0);
    unsigned bfs_samples = (int)(bfs_percentage * max_samples);
    vector<PartialAssignment> vk = {initial_state}, vk1 = {}; // depth k, depth k+1
    vector<int> pi_values = initial_state.get_values();
    PartialAssignment pi(initial_state, move(pi_values));
    pi.estimated_heuristic = initial_state.estimated_heuristic;
    pi.states_to_goal = initial_state.states_to_goal;
    shared_ptr<PartialAssignment> initial_state_ptr = make_shared<PartialAssignment>(pi);
    prepare_posprocessing(initial_state_ptr, task_proxy);
    vector<shared_ptr<PartialAssignment>> samples = {initial_state_ptr};
    leaves.push_back(initial_state);

    while ((samples.size() < bfs_samples && !vk.empty()) && !stopped) {
        rng->shuffle(vk);
        for (PartialAssignment& s : vk) {
            int idx_op = 0, rng_seed = (*rng)() * (INT32_MAX - 1);
            while (!stopped && samples.size() < bfs_samples) {
                OperatorID applied_op = OperatorID::no_operator;
                PartialAssignment s_ = dfss->sample_state_length(
                    s, rng_seed, idx_op, applied_op, is_valid_state
                );
                assert(
                    (idx_op == -1 && applied_op == OperatorID::no_operator) ||
                    (idx_op != -1 && applied_op != OperatorID::no_operator)
                );
                if (idx_op == -1)
                    break;

                if (allow_duplicates_intrarollout || hash_table.find(s_) == hash_table.end()) {
                    if (restart_h_when_goal_state && task_properties::is_goal_assignment(task_proxy, s_)) {
                        s_.estimated_heuristic = 0;
                        s_.states_to_goal = 0;
                    } else {
                        s_.estimated_heuristic = s.estimated_heuristic + (unit_cost ? 1 : operators[applied_op].get_cost());
                        s_.states_to_goal = s.states_to_goal + 1;
                    }
                    vector<int> s_values = s_.get_values();
                    PartialAssignment ps(s_, move(s_values));
                    ps.estimated_heuristic = s_.estimated_heuristic;
                    ps.states_to_goal = s_.states_to_goal;
                    shared_ptr<PartialAssignment> ps_ = make_shared<PartialAssignment>(ps);
                    if (prepare_posprocessing(ps_, task_proxy, true)) {
                        samples.push_back(ps_);
                        vk1.push_back(s_);
                        leaves.push_back(s_);
                        hash_table.insert(s_);
                    }
                }
                idx_op++;
                stopped = stop_sampling(true, bfs_percentage);
            }
            if (!stopped && samples.size() < bfs_samples)
                leaves.erase(find(leaves.begin(), leaves.end(), s));
            else
                break;
            stopped = stop_sampling(true, bfs_percentage);
        }
        vk = vk1;
        vk1.clear();
    }
    stopped = false; // reset to rw step
    return samples;
}

bool TechniqueGBackwardFsm::prepare_posprocessing(shared_ptr<PartialAssignment>& sample, const TaskProxy &task_proxy, bool from_bfs) {
    static unsigned num_samples = 0;

    if (from_bfs && avoid_bfs_core)
        bfs_core.insert(*sample);

    if (!sampling_engine::sui) {
        // Check hash conflict
        size_t h_key = sample->to_hash();
        if (sampleset.count(sample->values_to_string()) == 0) {
            sampleset.insert(sample->values_to_string());
            if (hashset.count(h_key) != 0)
                return false;
            hashset.insert(h_key);
        }

        // Maps sample successors
        PartialAssignment t(*sample);
        int op_cost;
        size_t s_key = sample->to_hash();
        for (const bool succ : { true, false }) { // true = generating successors of s; false = generating predecessor of s
            vector<OperatorID> applicable_operators;
            if (succ)
                succ_generator->generate_applicable_ops(*sample, applicable_operators, true);
            else
                predecessor_generator->generate_applicable_ops(*sample, applicable_operators, false);
            for (OperatorID& op_id : applicable_operators) {
                if (succ) {
                    OperatorProxy op_proxy = operators[op_id];
                    t = sample->get_partial_successor(op_proxy);
                    op_cost = op_proxy.get_cost();
                } else {
                    RegressionOperatorProxy op_proxy = regression_task_proxy->get_regression_operator(op_id);
                    assert(task_properties::is_applicable(op_proxy, *sample));
                    t = op_proxy.get_anonym_predecessor(*sample);
                    op_cost = op_proxy.get_cost();
                }
                if (!t.violates_mutexes()) {
                    vector<shared_ptr<PartialAssignment>> compatible_states;
                    trie.find_all_compatible(t.get_values(), SearchRule::supersets, compatible_states);
                    for (shared_ptr<PartialAssignment>& t_: compatible_states) {
                        size_t t_key = t_->to_hash();
                        if (t_key == s_key)
                            continue;
                        pair<size_t,int> pair = make_pair(succ ? t_key : s_key, op_cost);
                        if (find(sui_mapping[succ ? s_key : t_key].successors.begin(), sui_mapping[succ ? s_key : t_key].successors.end(), pair)
                                == sui_mapping[succ ? s_key : t_key].successors.end()) {
                            sui_mapping[succ ? s_key : t_key].successors.push_back(pair);
                        }
                    }
                }
            }
        }
        sui_mapping[s_key].samples.push_back(sample);
        if (sample->estimated_heuristic < sui_mapping[s_key].best_h)
            sui_mapping[s_key].best_h = sample->estimated_heuristic;

        // Add sample to trie
        trie.insert(sample->get_values(), sample);
    }

    int sample_step = (1 / sampling_technique::random_samples_percentage) - 1;
    if (!(num_samples % sample_step)) {
        create_random_samples(regression_task_proxy->get_goal_assignment(), 1, regression_depth_value + 1);
    }
    num_samples++;

    // State completion
    pair<bool,State> fs = sample->get_full_state(true, *rng);
    if (!fs.first) {
        utils::g_log << "[Sample Completion] Could not cast " << sample->to_binary(true)
            << " to full state. Undefined values will be output as 0 in binary." << endl;
    }
    sample->assign(fs.second.get_values());
    if (task_properties::is_goal_assignment(task_proxy, *sample))
        sample->estimated_heuristic = 0;

    return true;
}

void TechniqueGBackwardFsm::create_random_samples(PartialAssignment pa_aux, int num_random_samples, int random_sample_h) {
    if (num_random_samples == 0)
        return;

    const size_t n_atoms = pa_aux.get_values().size();
    while (num_random_samples > 0) {
        PartialAssignment random_sample(pa_aux, vector<int>(n_atoms, PartialAssignment::UNASSIGNED));
        pair<bool,State> p = random_sample.get_full_state(true, *rng);
        if (!p.first)
            continue;
        random_sample.assign(p.second.get_values());
        random_sample.estimated_heuristic = random_sample_h;
        sampling_technique::random_modified_tasks.push_back(make_shared<PartialAssignment>(random_sample));
        num_random_samples--;
    }
}

vector<shared_ptr<PartialAssignment>> TechniqueGBackwardFsm::create_next_all(
        shared_ptr<AbstractTask> seed_task, const TaskProxy &task_proxy) {
    auto t_sampling = std::chrono::high_resolution_clock::now();

    if (facts_file != "none") {
        utils::g_log << "[Sampling] Saving facts file in " << facts_file << endl;
        const AbstractTask *task = task_proxy.get_task();
        const vector<FactPair> relevant_facts = task_properties::get_strips_fact_pairs(task);
        ofstream out_file(facts_file);
        for (unsigned i = 0; i < relevant_facts.size(); i++) {
            out_file << task->get_fact_name(relevant_facts[i]);
            if (i < relevant_facts.size() - 1)
                out_file << ';';
        }
        out_file.close();
    }

    if (seed_task != last_task) {
        regression_task_proxy = make_shared<RegressionTaskProxy>(*seed_task);
        state_registry = make_shared<StateRegistry>(task_proxy);
        if (technique == "dfs" || technique == "bfs" || technique == "bfs_rw")
            dfss = make_shared<sampling::DFSSampler>(*regression_task_proxy, *rng);
        if (technique == "rw" || technique == "bfs_rw")
            rrws = make_shared<sampling::RandomRegressionWalkSampler>(*regression_task_proxy, *rng);
    }

    succ_generator = utils::make_unique_ptr<successor_generator::SuccessorGenerator>(task_proxy);
    predecessor_generator = utils::make_unique_ptr<predecessor_generator::PredecessorGenerator>(*regression_task_proxy);
    operators = task_proxy.get_operators();

    bias_reload_counter++;
    if (!bias_evaluator_tree.empty() &&
        (seed_task != last_task ||
         (bias_reload_frequency != -1 &&
          bias_reload_counter > bias_reload_frequency))) {
        options::OptionParser bias_parser(bias_evaluator_tree, *registry, *predefinitions, false);
        bias = bias_parser.start_parsing<shared_ptr<Heuristic>>();
        bias_reload_counter = 0;
        cache.clear();
    }

    PartialAssignmentBias *func_bias = nullptr;
    PartialAssignmentBias pab = [&](PartialAssignment &partial_assignment) {
        auto iter = cache.find(partial_assignment);
        if (iter != cache.end()) {
            return iter->second;
        } else {
            int h = compute_heuristic(task_proxy, bias.get(), *rng, partial_assignment);
            cache[partial_assignment] = h;
            return h;
        }
    };

    auto is_valid_state = [&](PartialAssignment &partial_assignment) {
        if (state_filtering == "none") {
            return true;
        } else if (state_filtering == "mutex") {
            return !(is_valid_walk) || regression_task_proxy->convert_to_full_state(partial_assignment, true, *rng).first;
        } else {
            utils::g_log << "[ERROR] Unknown state_filtering: " << state_filtering << endl;
            exit(0);
        }
        return false;
    };

    if (bias != nullptr) {
        func_bias = &pab;
    }

    if (allow_duplicates_interrollout)
        hash_table.clear();

    PartialAssignment pa = regression_task_proxy->get_goal_assignment();
    pa.estimated_heuristic = 0;
    vector<shared_ptr<PartialAssignment>> samples;
    vector<PartialAssignment> leaves;

    if (samples_per_search == -1)
        samples_per_search = max_samples;

    float regression_depth_n = -1;
    if (is_number(regression_depth)) {
        regression_depth_n = stoi(regression_depth);
    } else {
        if (regression_depth == "default") {
            if (technique == "dfs" || technique == "bfs")
                regression_depth_n = depth_k;
            else // rw, bfs_rw
                regression_depth_n = samples_per_search;
        } else if (regression_depth == "facts") {
            regression_depth_n = pa.to_binary().length();
        } else if (regression_depth == "facts_per_avg_effects") {
            int num_props = pa.to_binary().length();
            int num_effects = 0, num_ops = 0;
            for (OperatorProxy op : operators) {
                num_ops++;
                for (EffectProxy eff : op.get_effects()) {
                    num_effects++;
                }
            }
            float mean_num_effects = (float)num_effects / num_ops;
            regression_depth_n = (float)num_props / mean_num_effects;
        } else {
            utils::g_log << "[ERROR] Unknown regression_depth: " << regression_depth << endl;
            exit(0);
        }
    }
    assert(regression_depth_n > 0);
    regression_depth_value = ceil(regression_depth_multiplier * regression_depth_n);

    if (technique == "rw" || technique == "bfs_rw") {
        samples_per_search = ceil(regression_depth_multiplier * regression_depth_n);
    } else if (technique == "dfs" || technique == "bfs") {
        depth_k = ceil(regression_depth_multiplier * regression_depth_n);
    }

    static bool first_call = true;
    if (first_call) {
        utils::g_log << "[Sampling] Starting the sampling (algorithm " << technique << ")..." << endl;
        utils::g_log << "[Sampling] State filtering: " << state_filtering << endl;
        utils::g_log << "[Sampling] Regression depth value: " << regression_depth_value << endl;
        first_call = false;
    }

    if (technique == "rw") {
        samples = sample_with_random_walk(pa, samples_per_search, is_valid_state, func_bias, task_proxy);
        utils::g_log << "[Sampling] RW rollout sampled " << samples.size() << " states." << endl;        

    } else if (technique == "bfs_rw") {
        utils::g_log << "[Sampling] Starting BFS step..." << endl;
        samples = sample_with_percentage_limited_bfs(bfs_percentage, pa, is_valid_state, leaves, task_proxy);
        utils::g_log << "[Sampling] BFS step sampled " << samples.size() << " states." << endl;

    } else if (technique == "dfs" || technique == "bfs") {
        do {
            vector<shared_ptr<PartialAssignment>> samples_ = sample_with_bfs_or_dfs(
                technique, pa, max_samples-samples.size(), is_valid_state, task_proxy
            );
            utils::g_log << "[Sampling] " << (technique == "bfs" ? "BFS" : "DFS")
                << " rollout sampled " << samples.size() << " states. "
                << "Looking for " << (unsigned)max_samples-samples.size() << " more." << endl;
            samples.insert(samples.end(), samples_.begin(), samples_.end());
            if (allow_duplicates_interrollout)
                hash_table.clear();
        } while ((samples.size() < (unsigned)max_samples) && !stopped);

    } else {
        utils::g_log << "[ERROR] " << technique << " not implemented!" << endl;
        exit(0);
    }

    if (technique == "bfs_rw") {
        // bfs_rw random walk step
        if (leaves.size() <= 0) {
            utils::g_log << "[Sampling] The whole statespace was sampled. Skipping RW step." << endl;
            stopped = true;
            return samples;
        }

        utils::g_log << "[Sampling] Starting RW from " << leaves.size() << " leaves" << endl;
        if (max_samples != numeric_limits<int>::max())
            utils::g_log << "[Sampling] Looking for " << (max_samples - samples.size()) << " more samples..." << endl;
        else
            utils::g_log << "[Sampling] Looking for more samples until mem/time budget runs out." << endl;

        float pct_for_next_print = bfs_percentage;
        int total_rw_samples = 0, total_rollouts = 0;
        int lid = 0;
        vector<bool> leaves_used(leaves.size(), false);
        while ((samples.size() < (unsigned)max_samples) && !stopped) {
            do {
                lid = (int)((*rng)() * (INT32_MAX - 1)) % leaves.size();
            } while (leaves_used[lid]);
            leaves_used[lid] = true;
            if (all_of(leaves_used.begin(), leaves_used.end(), [](bool v) {return v;}))
                fill(leaves_used.begin(), leaves_used.end(), false);

            vector<shared_ptr<PartialAssignment>> samples_ = sample_with_random_walk(
                leaves[lid],
                min(samples_per_search - leaves[lid].states_to_goal, (int)(max_samples-samples.size())),
                is_valid_state,
                func_bias,
                task_proxy,
                false,
                !allow_duplicates_interrollout,
                bfs_core
            );
            samples.insert(samples.end(), samples_.begin(), samples_.end());
            total_rw_samples += samples_.size();
            total_rollouts++;
            if ((float)samples.size() / max_samples > pct_for_next_print || samples.size() >= (unsigned)max_samples) {
                pct_for_next_print += 0.1;
                utils::g_log << "[Sampling] RW: " << total_rw_samples << " states sampled in " << total_rollouts
                    << " rollouts (avg: " << ((float)total_rw_samples / total_rollouts) << ")" << endl;
            }
        }
    }

    if (technique != "rw") {
        utils::g_log << "[Sampling] Done in " << fixed << (std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_sampling).count() / 1000.0) << "s." << endl;

        hashset.clear();
        sampleset.clear();
    }

    return samples;
}

/* PARSING TECHNIQUE_GBACKWARD_FSM*/
static shared_ptr<TechniqueGBackwardFsm> _parse_technique_gbackward_fsm(
        options::OptionParser &parser) {
    SamplingTechnique::add_options_to_parser(parser);
    parser.add_option<string>(
            "technique",
            "Search technique (rw, dfs, bfs, bfs_rw). "
            "If bfs_rw then set bfs_percentage.",
            "rw"
    );
    parser.add_option<string>(
            "regression_depth",
            "How to bound each rollout: 'default', 'facts', 'facts_per_avg_effects', int",
            "default"
    );
    parser.add_option<int>(
            "depth_k",
            "Maximum depth using the dfs/bfs algorithm. "
            "If it doesn't reach max_samples, complete with random walks of each leaf state.",
            "99999"
    );
    parser.add_option<string>(
            "allow_duplicates",
            "Allow sample duplicated states in [all, interrollout, none]",
            "interrollout"
    );
    parser.add_option<bool>(
            "wrap_partial_assignment",
            "If set, wraps a partial assignment obtained by the regression for the "
            "initial state into a task which has additional values for undefined "
            "variables. By default, the undefined variables are random uniformly "
            "set (satisfying the mutexes).",
            "false"
    );
    parser.add_option<bool>(
            "deprioritize_undoing_steps",
            "Deprioritizes actions which undo the previous action",
            "false"
    );
    parser.add_option<bool>(
            "is_valid_walk",
            "enforces states during random walk are valid states w.r.t. "
            "the KNOWN mutexes",
            "true"
    );
    parser.add_option<bool>(
            "restart_h_when_goal_state",
            "Restart h value when goal state is sampled.",
            "true"
    );
    parser.add_option<string>(
            "state_filtering",
            "Filtering of applicable operators (none or mutex)",
            "mutex"
    );
    parser.add_option<double>(
            "bfs_percentage",
            "Percentage of samples per BFS when technique=bfs_rw",
            "0.1"
    );
    parser.add_option<shared_ptr<Heuristic>>(
            "bias",
            "bias heuristic",
            "<none>"
    );
    parser.add_option<int>(
            "bias_reload_frequency",
            "the bias is reloaded everytime the tasks for which state are"
            "generated changes or if it has not been reloaded for "
            "bias_reload_frequency steps. Use -1 to prevent reloading.",
            "-1"
    );
    parser.add_option<bool>(
            "bias_probabilistic",
            "uses the bias values as weights for selecting the next state"
            "on the walk. Otherwise selects a random state among those with "
            "maximum bias",
            "true"
    );
    parser.add_option<double>(
            "bias_adapt",
            "if using the probabilistic bias, then the bias values calculated"
            "for the successors s1,..., sn of the state s are adapted as "
            "bias_adapt^(b(s1) - b(s)). This gets right of the issue that for"
            "large bias values, there was close to no difference between the "
            "states probabilities and focuses more on states increasing the bias.",
            "-1"
    );
    parser.add_option<int>(
            "max_upgrades",
            "Maximum number of times this sampling technique can upgrade its"
            "parameters. Use -1 for infinite times.",
            "0"
    );
    parser.add_option<string>(
            "facts_file",
            "Path to save facts file.",
            "none"
    );
    options::Options opts = parser.parse();

    shared_ptr<TechniqueGBackwardFsm> technique;
    if (!parser.dry_run()) {
        technique = make_shared<TechniqueGBackwardFsm>(opts);
    }
    return technique;
}

static Plugin<SamplingTechnique> _plugin_technique_gbackward_fsm(
        TechniqueGBackwardFsm::name, _parse_technique_gbackward_fsm);
}
