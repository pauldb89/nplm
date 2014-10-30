#pragma once

namespace nplm {

/**
 * Wraps the feature values computed from the LM language model.
 */
struct LMFeatures {
  LMFeatures();

  LMFeatures(double lm_score, double oov_score);

  LMFeatures& operator+=(const LMFeatures& other);

  double LMScore;
  double OOVScore;
};

} // namespace nplm
