#pragma once

#include "util.h"

namespace nplm {

class ContextExtractor {
 public:
  ContextExtractor(
      const shared_ptr<Corpus>& corpus,
      int context_size, int sos_id, int eos_id);

  // Extracts the full context (including the current word) for the given index.
  // Format: [w_{i-n+1}, ..., w_{i-1}, w_i]
  VectorInt extract(data_size_t index) const;

 private:
  shared_ptr<Corpus> corpus;
  int contextSize, sosId, eosId;
};

} // namespace nplm
