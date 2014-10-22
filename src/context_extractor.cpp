#include "context_extractor.h"

namespace nplm {

ContextExtractor::ContextExtractor(
    const shared_ptr<Corpus>& corpus, int context_size, int sos_id, int eos_id)
    : corpus(corpus), contextSize(context_size), sosId(sos_id), eosId(eos_id) {}

VectorInt ContextExtractor::extract(data_size_t index) const {
  vector<int> context(contextSize);
  context[0] = corpus->at(index);

  bool sentence_start = false;
  for (int i = 1; i < contextSize; ++i) {
    data_size_t current_index = index - i;
    sentence_start |= current_index < 0 || corpus->at(current_index) == eosId;
    context[i] = sentence_start ? sosId : corpus->at(current_index);
  }

  reverse(context.begin(), context.end());

  VectorInt result = VectorInt::Zero(contextSize);
  for (int i = 0; i < contextSize; ++i) {
    result(i) = context[i];
  }

  return result;
}

} // namespace nplm
