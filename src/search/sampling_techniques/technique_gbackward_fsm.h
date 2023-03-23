#ifndef SAMPLING_TECHNIQUES_TECHNIQUE_GBACKWARD_FSM_H
#define SAMPLING_TECHNIQUES_TECHNIQUE_GBACKWARD_FSM_H

#include <unordered_map>
#include "../trie/trie.h"

#include "sampling_technique.h"
#include "../task_proxy.h"
#include "../utils/distribution.h"
#include "../task_utils/sampling.h"
#include "../task_utils/successor_generator.h"
#include "../task_utils/predecessor_generator.h"

class StateRegistry;
class RegressionTaskProxy;
namespace sampling {
class RandomRegressionWalkSampler;
class DFSSampler;
}

using namespace std;

namespace sampling_technique {
class TechniqueGBackwardFsm : public SamplingTechnique {
protected:
    const string technique;
    const string regression_depth;
    int depth_k;
    const bool allow_duplicates_interrollout;
    const bool allow_duplicates_intrarollout;
    const bool wrap_partial_assignment;
    const bool deprioritize_undoing_steps;
    const bool is_valid_walk;
    const bool restart_h_when_goal_state;
    const string state_filtering;
    const double bfs_percentage;
    const options::ParseTree bias_evaluator_tree;
    const bool bias_probabilistic;
    const double bias_adapt;
    utils::HashMap<PartialAssignment, int> cache;
    shared_ptr<Heuristic> bias = nullptr;
    const int bias_reload_frequency;
    int bias_reload_counter;
    const string facts_file;
    shared_ptr<StateRegistry> state_registry = nullptr;
    shared_ptr<AbstractTask> last_partial_wrap_task = nullptr;
    shared_ptr<RegressionTaskProxy> regression_task_proxy = nullptr;
    shared_ptr<sampling::RandomRegressionWalkSampler> rrws = nullptr;
    shared_ptr<sampling::DFSSampler> dfss = nullptr;
    utils::HashSet<PartialAssignment> hash_table;
    std::unique_ptr<successor_generator::SuccessorGenerator> succ_generator;
    std::unique_ptr<predecessor_generator::PredecessorGenerator> predecessor_generator;
    OperatorsProxy operators;
    utils::HashSet<PartialAssignment> bfs_core;
    bool avoid_bfs_core = true;

    unordered_set<size_t> hashset;
    unordered_set<string> sampleset;

    virtual vector<shared_ptr<PartialAssignment>> create_next_all(
            shared_ptr<AbstractTask> seed_task,
            const TaskProxy &task_proxy) override;

    // virtual void do_upgrade_parameters() override ;

public:
    explicit TechniqueGBackwardFsm(const options::Options &opts);
    virtual ~TechniqueGBackwardFsm() override = default;

    // virtual void dump_upgradable_parameters(ostream &/*stream*/) const override;

    virtual const string &get_name() const override;
    const static string name;

private:
    vector<shared_ptr<PartialAssignment>> sample_with_random_walk(
        PartialAssignment initial_state,
        const unsigned steps,
        const ValidStateDetector &is_valid_state,
        const PartialAssignmentBias *bias,
        const TaskProxy &task_proxy,
        const bool sample_initial_state = true,
        const bool global_hash_table = true,
        const utils::HashSet<PartialAssignment> states_to_avoid = utils::HashSet<PartialAssignment>()
    );

    vector<shared_ptr<PartialAssignment>> sample_with_bfs_or_dfs(
        string technique,
        PartialAssignment initial_state,
        const unsigned steps,
        const ValidStateDetector &is_valid_state,
        const TaskProxy &task_proxy
    );

    vector<shared_ptr<PartialAssignment>> sample_with_percentage_limited_bfs(
        double bfs_percentage,
        PartialAssignment initial_state,
        const ValidStateDetector &is_valid_state,
        vector<PartialAssignment> &leaves,
        const TaskProxy &task_proxy
    );

    bool prepare_posprocessing(
        shared_ptr<PartialAssignment>& sample,
        const TaskProxy &task_proxy,
        bool from_bfs = false
    );

    void create_random_samples(
        PartialAssignment pa_aux,
        int num_random_samples,
        int random_sample_h
    );
};
}
#endif
