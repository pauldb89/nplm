#pragma once

#include <memory>
#include <string>

#include "cdec_nplm_mapper.h"
#include "cdec_rule_converter.h"
#include "cdec_state_converter.h"
#include "lm_features.h"
#include "model.h"
#include "query_cache.h"
#include "neuralLM.h"

#include "ff.h"

struct SentenceMetadata;

namespace HG {
  struct Edge;
};

namespace nplm {

class CdecNPLMWrapper : public FeatureFunction {
 public:
  CdecNPLMWrapper(
      const string& filename,
      const string& feature_name,
      bool normalized,
      bool persistent_cache);

  virtual void PrepareForInput(const SentenceMetadata& smeta);

  ~CdecNPLMWrapper();

 protected:
  virtual void TraversalFeaturesImpl(
      const SentenceMetadata& smeta, const HG::Edge& edge,
      const vector<const void*>& prev_states, SparseVector<double>* features,
      SparseVector<double>* estimated_features, void* next_state) const;

  virtual void FinalTraversalFeatures(
      const void* prev_state, SparseVector<double>* features) const;

 private:
  void savePersistentCache();

  void loadPersistentCache(int sentence_id);

  LMFeatures scoreFullContexts(const vector<int>& symbols) const;

  double getScore(const vector<int>& ngram) const;

  LMFeatures scoreContext(const vector<int>& symbols, int position) const;

  void constructNextState(const vector<int>& symbols, void* state) const;

  LMFeatures estimateScore(const vector<int>& symbols) const;

  int fid;
  int fidOOV;
  int contextWidth;
  string filename;

  shared_ptr<Vocabulary> vocab;
  shared_ptr<NeuralLM> model;
  shared_ptr<CdecNPLMMapper> mapper;
  shared_ptr<CdecRuleConverter> ruleConverter;
  shared_ptr<CdecStateConverter> stateConverter;

  int kSTART;
  int kSTOP;
  int kUNKNOWN;
  int kSTAR;

  bool normalized;

  bool persistentCache;
  string cacheFile;
  mutable QueryCache cache;
  mutable int cacheHits, totalHits;
};

} // namespace nplm
