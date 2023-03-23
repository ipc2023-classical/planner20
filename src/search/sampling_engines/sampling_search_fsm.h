#ifndef SEARCH_ENGINES_SAMPLING_SEARCH_FSM_H
#define SEARCH_ENGINES_SAMPLING_SEARCH_FSM_H

#include "sampling_search_base.h"

#include <vector>
#include <random>

#include <unordered_map>
#include <unordered_set>

namespace options {
class Options;
}

namespace sampling_engine {

class SamplingSearchFsm : public SamplingSearchBase {
protected:
    const bool store_plan_cost;
    const bool store_state;
    const bool sai_partial;
    const bool sai_complete;
    const int random_multiplier;
    const std::vector<FactPair> relevant_facts;
    StateRegistry registry;
    std::string header;
    std::shared_ptr<utils::RandomNumberGenerator> rng;

    virtual std::vector<std::string> extract_samples() override;
    virtual std::string construct_header() const;
    virtual std::string sample_file_header() const override;

public:
    explicit SamplingSearchFsm(const options::Options &opts);
    virtual ~SamplingSearchFsm() override = default;

private:
    std::vector<std::string> format_output(std::vector<std::shared_ptr<PartialAssignment>>& samples);
    void successor_improvement();
    void sample_improvement(std::vector<std::shared_ptr<PartialAssignment>>& samples);
};
}
#endif
