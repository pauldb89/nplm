#include "query_cache.h"

namespace nplm {

pair<double, bool> QueryCache::get(const vector<int>& query) const {
  auto it = cache.find(query);
  return it == cache.end() ? make_pair(double(0), false) : make_pair(it->second, true);
}

void QueryCache::put(const vector<int>& query, double value) {
  cache[query] = value;
}

size_t QueryCache::size() const {
  return cache.size();
}

void QueryCache::clear() {
  cache.clear();
}

bool QueryCache::operator==(const QueryCache& other) const {
  return cache == other.cache;
}

} // namespace nplm
