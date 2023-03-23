#ifndef OPTIONS_OPTIONS_H
#define OPTIONS_OPTIONS_H

#include "any.h"
#include "errors.h"
#include "type_namer.h"

#include "../utils/system.h"

#include <string>
#include <typeinfo>
#include <unordered_map>

namespace options {
class Registry;
class Predefinitions;

// Wrapper for unordered_map<string, Any>.
class Options {
    std::unordered_map<std::string, Any> storage;
    std::unordered_map<std::string, ParseTree> parse_trees;
    std::string unparsed_config;
    const bool help_mode;
    Registry *registry = nullptr;
    const Predefinitions *predefinitions = nullptr;

public:
    explicit Options(bool help_mode = false);

    template<typename T>
    void set(const std::string &key, T value) {
        storage[key] = value;
    }

    template<typename T>
    T get(const std::string &key) const {
        const auto it = storage.find(key);
        if (it == storage.end()) {
            ABORT_WITH_DEMANGLING_HINT(
                "Attempt to retrieve nonexisting object of name " + key +
                " (type: " + typeid(T).name() + ")", typeid(T).name());
        }
        try {
            T result = any_cast<T>(it->second);
            return result;
        } catch (const BadAnyCast &) {
            ABORT_WITH_DEMANGLING_HINT(
                "Invalid conversion while retrieving config options!\n" +
                key + " is not of type " + typeid(T).name(), typeid(T).name());
        }
    }

    template<typename T>
    T get(const std::string &key, const T &default_value) const {
        if (storage.count(key))
            return get<T>(key);
        else
            return default_value;
    }
    // TODO pat maybe make movable
    void set_parse_tree(const std::string &key, const ParseTree parse_tree) {
        parse_trees[key] = parse_tree;
    }

    const ParseTree &get_parse_tree(const std::string &key) const {
        const auto it = parse_trees.find(key);
        if (it == parse_trees.end()) {
            ABORT("Attempt to retrieve nonexisting parse_tree of name " + key);
        }
        return it->second;
    }

    const ParseTree &get_parse_tree(const std::string &key, const ParseTree &default_value) const {
        const auto it = parse_trees.find(key);
        if (it == parse_trees.end()) {
            return default_value;
        }
        return it->second;
    }

    template<typename T>
    void verify_list_non_empty(const std::string &key) const {
        if (!help_mode) {
            if (get_list<T>(key).empty()) {
                throw OptionParserError("Error: list for key " +
                                        key + " must not be empty\n");
            }
        }
    }

    template<typename T>
    std::vector<T> get_list(const std::string &key) const {
        return get<std::vector<T>>(key);
    }

    bool contains(const std::string &key) const;
    const std::string &get_unparsed_config() const;
    void set_unparsed_config(const std::string &config);
    void set_registry(Registry *registry_);
    Registry *get_registry() const;
    void set_predefinitions(const Predefinitions *predefinitions_);
    const Predefinitions *get_predefinitions() const;
};
}

#endif