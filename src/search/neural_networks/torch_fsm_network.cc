#include "torch_fsm_network.h"

#include "abstract_network.h"

#include "../option_parser.h"
#include "../plugin.h"

#include "../task_utils/task_properties.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>

using namespace std;

vector<string> get_facts(vector<string> facts) {
    if (facts.size() == 1 && facts[0].rfind("file ", 0) == 0){
        vector<string> loaded_facts;
        ifstream myfile (facts[0].substr(5));
        cout << facts[0].substr(5) << endl;
        string line;
        if (myfile.is_open()) {
            while (getline (myfile,line, ';')) {
                if (line[line.size()-1] == '\n')
                    line.pop_back();
                cout << line << endl;
                loaded_facts.push_back(line);
            }
            myfile.close();
        }
        return loaded_facts;
    } else {
        return facts;
    }
}

vector<string> get_defaults(vector<string> defaults) {
    if (defaults.size() == 1 && defaults[0].rfind("file ", 0) == 0){
        vector<string> loaded_defaults;
        ifstream myfile (defaults[0].substr(5));
        string line;
        if (myfile.is_open()) {
            while (getline (myfile,line, ';')) {
                loaded_defaults.push_back(line);
            }
            myfile.close();
        }
        return loaded_defaults;
    } else {
        return defaults;
    }
}

namespace neural_networks {

TorchFsmNetwork::TorchFsmNetwork(const Options &opts)
    : TorchNetwork(opts),
      heuristic_shift(opts.get<int>("shift")),
      heuristic_multiplier(opts.get<int>("multiplier")),
      relevant_facts(get_fact_mapping(task.get(), get_facts(opts.get_list<string>("facts")))),
      default_input_values(get_default_inputs(get_defaults(opts.get_list<string>("defaults")))),
      unary_threshold(opts.get<double>("unary_threshold")),
      undefined_input(opts.get<bool>("undefined_input")) {
    check_facts_and_default_inputs(relevant_facts, default_input_values);

    // Check if output is normalized
    normalize_output = false;
    string path = opts.get<string>("path");
    string train_args_file = path.substr(0, path.find("/model")) + "/train_args.json";
    ifstream f(train_args_file);
    string line;
    if (f.is_open()) {
        while(getline(f, line)) {
            if (line.find("\"normalize_output\": true") != string::npos)
                normalize_output = true;
            if (line.find("\"max_h\": ") != string::npos) {
                max_h = stoi(line.substr(
                    line.find(": ") + 2,
                    line[line.size()-1] == ',' ? line.size()-1 : line.size()-2)
                );
            }
        }
        f.close();
    }
}

TorchFsmNetwork::~TorchFsmNetwork() {}


bool TorchFsmNetwork::is_heuristic() {
    return true;
}

int TorchFsmNetwork::get_heuristic() {
    return last_h;
}

int TorchFsmNetwork::unary_to_value(const vector<double>& unary_h) {
    for (size_t i = 0; i < unary_h.size(); i++) {
        if (unary_h[i] < unary_threshold)
            return i - 1;
    }
    return unary_h.size() - 1;
}

const vector<int> &TorchFsmNetwork::get_heuristics() {
    return last_h_batch;
}

vector<at::Tensor> TorchFsmNetwork::get_input_tensors(const State &state) {
    state.unpack();
    const vector<int> &values = state.get_values();

    unsigned size = relevant_facts.size();
    if (undefined_input)
        size += relevant_facts[relevant_facts.size()-1].var + 1;
    at::Tensor tensor = torch::ones({1, static_cast<long>(size)});
    auto accessor = tensor.accessor<float, 2>();

    size_t idx = 0;
    for (unsigned i = 0; i < relevant_facts.size(); i++) {
        if (relevant_facts[i] == FactPair::no_fact) {
            accessor[0][idx] = default_input_values[idx];
        } else {
            if (undefined_input && (i == 0 || relevant_facts[i].var != relevant_facts[i-1].var))
                accessor[0][idx++] = values[relevant_facts[i].var] == PartialAssignment::UNASSIGNED;
            accessor[0][idx] = values[relevant_facts[i].var] == relevant_facts[i].value;
        }
        idx++;
    }
    return {tensor};
}

int TorchFsmNetwork::h_adjustment(int h) {
    return (h + heuristic_shift) * heuristic_multiplier;
}

void TorchFsmNetwork::parse_output(const torch::jit::IValue &output) {
    at::Tensor tensor = output.toTensor();
    std::vector<double> unary_output(tensor.data_ptr<float>(),
                                    tensor.data_ptr<float>() + tensor.numel());

    // Regression
    if (tensor.size(1) == 1) {
        last_h = round(normalize_output ? unary_output[0] * max_h : unary_output[0]);
    // Classification
    } else {
        last_h = unary_to_value(unary_output);
    }

    last_h_batch.push_back(h_adjustment(last_h));
}

void TorchFsmNetwork::clear_output() {
    last_h = Heuristic::NO_VALUE;
    last_h_batch.clear();
}
}

static shared_ptr<neural_networks::AbstractNetwork> _parse(OptionParser &parser) {
    parser.document_synopsis(
        "Torch FSM Network",
        "Takes a trained PyTorch model and evaluates it on a given state."
        "The output is read as and provided as a heuristic.");
    neural_networks::TorchNetwork::add_options_to_parser(parser);
    parser.add_option<int>(
            "shift",
            "shift the predicted heuristic value (useful, if the model"
            "output is expected to be negative up to a certain bound.",
            "0");
    parser.add_option<int>(
            "multiplier",
            "Multiply the predicted (and shifted) heuristic value (useful, if "
            "the model predicts small float values, but heuristics have to be "
            "integers",
            "1");
    parser.add_list_option<string>(
            "facts",
            "if the SAS facts after translation can differ from the facts"
            "during training (e.g. some are pruned or their order changed),"
            "provide here the order of facts during training.",
            "[]");
    parser.add_list_option<string>(
            "defaults",
            "Default values for the facts given in option 'facts'",
            "[]");
    parser.add_option<double>(
            "unary_threshold",
            "Threshold to use when interpreting unary heuristic values to a single value.",
            "0.01");
    parser.add_option<bool>(
            "undefined_input",
            "Add neurons to undefined facts on input.",
            "false");

    Options opts = parser.parse();

    shared_ptr<neural_networks::TorchFsmNetwork> network;
    if (!parser.dry_run()) {
        network = make_shared<neural_networks::TorchFsmNetwork>(opts);
    }

    return network;
}

static Plugin<neural_networks::AbstractNetwork> _plugin("torch_fsm_network", _parse);
