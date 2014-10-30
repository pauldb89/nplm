#pragma once

#include <unordered_map>

#include <boost/functional/hash.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include "serialization_helpers.h"

using namespace std;

namespace nplm {

class QueryCache {
 public:
  pair<double, bool> get(const vector<int>& query) const;

  void put(const vector<int>& query, double value);

  size_t size() const;

  void clear();

  bool operator==(const QueryCache& other) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & cache;
  }

 public:
  unordered_map<vector<int>, double, boost::hash<vector<int>>> cache;
};

} // namespace nplm
