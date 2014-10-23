#include "minibatch_extractor.h"

namespace nplm {

MinibatchExtractor::MinibatchExtractor(
    const shared_ptr<Corpus>& corpus,
    const shared_ptr<Vocabulary>& vocab,
    const Config& config)
    : config(config) {
  extractor = make_shared<ContextExtractor>(
      corpus, config.ngram_size, vocab->lookup_word("<s>"),
      vocab->lookup_word("</s>"));
  indices.resize(corpus->size());
  iota(indices.begin(), indices.end(), 0);
  random_shuffle(indices.begin(), indices.end());
}

MatrixInt MinibatchExtractor::extract(data_size_t start_index) const {
  data_size_t actual_size = min(
      static_cast<data_size_t>(indices.size()) - start_index,
      static_cast<data_size_t>(config.minibatch_size));

  MatrixInt minibatch(config.ngram_size, actual_size);
  for (data_size_t i = 0; i < actual_size; ++i) {
    minibatch.col(i) = extractor->extract(indices[start_index + i]);
  }

  return minibatch;
}

} // namespace nplm
