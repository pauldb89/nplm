#pragma once

#include <vector>

#include "config.h"
#include "context_extractor.h"
#include "vocabulary.h"

using namespace std;

namespace nplm {

class MinibatchExtractor {
 public:
  MinibatchExtractor(
      const shared_ptr<Corpus>& corpus,
      const shared_ptr<Vocabulary>& vocab,
      const Config& config);

  MatrixInt extract(data_size_t start_index) const;

 private:
  shared_ptr<ContextExtractor> extractor;
  Config config;
  vector<data_size_t> indices;
};

} // namespace nplm
