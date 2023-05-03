#ifndef SAMPLING_TECHNIQUES_SAMPLING_TECHNIQUE_H
#define SAMPLING_TECHNIQUES_SAMPLING_TECHNIQUE_H

#include "../options/parse_tree.h"

#include "../tasks/root_task.h"

#include "../utils/hash.h"
#include "../utils/rng_options.h"
#include "../utils/countdown_timer.h"
#include "../utils/memory.h"
#include "../ext/trie.h"

#include <memory>
#include <ostream>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <limits>

class AbstractTask;
class GoalsProxy;
class Heuristic;
class PartialAssignment;
class State;
class TaskProxy;

namespace options {
class OptionParser;
class Options;
class Predefinitions;
class Registry;
}

namespace sampling_technique {
extern std::shared_ptr<AbstractTask> modified_task;
extern std::vector<std::shared_ptr<PartialAssignment>> modified_tasks;
extern std::vector<std::shared_ptr<PartialAssignment>> random_modified_tasks;

class SuiNode {
public:
    std::vector<std::shared_ptr<PartialAssignment>> samples;
    std::vector<std::pair<size_t,int>> successors;
    int best_h = std::numeric_limits<int>::max();
};

class SuiNodePtrCompare {
public:
    bool operator()(SuiNode* const first, SuiNode* const second) {
        return first->best_h > second->best_h;
    };
};

extern int random_samples;
extern float random_samples_percentage;
extern std::string random_estimates;
extern trie::trie<std::shared_ptr<PartialAssignment>> trie;
extern std::unordered_map<size_t,SuiNode> sui_mapping;
extern double max_time;

class SamplingTechnique {
private:
    static int next_id;
public:
    const int id;
    int regression_depth_value = 0;
protected:
    options::Registry *registry;
    const options::Predefinitions *predefinitions;
    int searches;
    int samples_per_search;
    int max_samples;
    bool limit_reached;
public:
    const bool unit_cost;
protected:
    double random_percentage;
    std::string random_estimates;
    const double regression_depth_multiplier;
    double max_time;
    const int mem_limit_mb;
    const bool remove_duplicates;
    const bool check_mutexes;
    const bool check_solvable;
    const bool use_alternative_mutexes;
    const std::vector<std::vector<std::string>> alternative_mutexes;
    const options::ParseTree eval_parse_tree;
    const std::unique_ptr<options::OptionParser> option_parser;
    std::unique_ptr<utils::CountdownTimer> sampling_timer;
    int mem_limit;
    int mem_presampling = utils::get_peak_memory_in_kb();
    int counter = 0;
    bool stopped = false;

protected:
    int remaining_upgrades;

    std::shared_ptr<utils::RandomNumberGenerator> rng;
    std::vector<std::vector<std::set<FactPair>>> alternative_task_mutexes;
    std::shared_ptr<AbstractTask> last_task = nullptr;

    virtual std::shared_ptr<AbstractTask> create_next(
        std::shared_ptr<AbstractTask> /*seed_task*/,
        const TaskProxy &/*task_proxy*/) { return {}; };
    virtual std::vector<std::shared_ptr<PartialAssignment>> create_next_all(
        std::shared_ptr<AbstractTask> /*seed_task*/,
        const TaskProxy &/*task_proxy*/) { return {}; };

    bool test_mutexes(const std::shared_ptr<AbstractTask> &task) const;
    bool test_solvable(const TaskProxy &task_proxy) const;
//    void dump_modifications(std::shared_ptr<AbstractTask> task) const;
    void update_alternative_task_mutexes(const std::shared_ptr<AbstractTask> &task);

    virtual void do_upgrade_parameters();

public:
    explicit SamplingTechnique(const options::Options &opts);
    SamplingTechnique(int searches,
//                      std::string dump_directory,
                      bool check_mutexes,
                      bool check_solvable, std::mt19937 &mt = utils::get_global_mt19937());
    virtual ~SamplingTechnique();

    int get_count() const;
    int get_counter() const;
    bool empty() const;
    bool stop_sampling(bool is_bfs = false, float bfs_pct = 0.1) const;
    int mem_usage_mb() const;

    std::shared_ptr<AbstractTask> next(
        const std::shared_ptr<AbstractTask> &seed_task = tasks::g_root_task);
    std::shared_ptr<AbstractTask> next(
        const std::shared_ptr<AbstractTask> &seed_task,
        const TaskProxy &task_proxy);
    std::vector<std::shared_ptr<PartialAssignment>> next_all(
        const std::shared_ptr<AbstractTask> &seed_task = tasks::g_root_task);

    virtual void initialize() {
    }
    virtual const std::string &get_name() const = 0;

    static void add_options_to_parser(options::OptionParser &parser);
    static std::vector<int> extractInitialState(const State &state);
    static std::vector<FactPair> extractGoalFacts(const GoalsProxy &goals_proxy);
    static std::vector<FactPair> extractGoalFacts(const State &state);

    virtual bool has_upgradeable_parameters() const;
    virtual void upgrade_parameters();
    virtual void dump_upgradable_parameters(std::ostream &stream) const;

    static const std::string no_dump_directory;
};
}

#endif
