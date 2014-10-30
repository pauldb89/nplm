#include "lm_features.h"

namespace nplm {

LMFeatures::LMFeatures() : LMScore(0), OOVScore(0) {}

LMFeatures::LMFeatures(double lm_score, double oov_score)
    : LMScore(lm_score), OOVScore(oov_score) {}

LMFeatures& LMFeatures::operator+=(const LMFeatures& other) {
  LMScore += other.LMScore;
  OOVScore += other.OOVScore;
  return *this;
}

} // namespace nplm
