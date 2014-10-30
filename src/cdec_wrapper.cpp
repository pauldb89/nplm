#include "cdec_wrapper.h"

#include <iostream>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/filesystem.hpp>

// cdec headers
#include "hg.h"
#include "sentence_metadata.h"

using namespace std;
using namespace nplm;

namespace nplm {

CdecNPLMWrapper::CdecNPLMWrapper(
    const string& filename,
    const string& feature_name,
    bool normalized,
    bool persistent_cache)
    : fid(FD::Convert(feature_name)),
      fidOOV(FD::Convert(feature_name + "_OOV")),
      filename(filename),
      persistentCache(persistent_cache), cacheHits(0), totalHits(0) {
  omp_set_num_threads(1);
  Eigen::setNbThreads(1);

  model = make_shared<NeuralLM>(filename, true);
  model->set_normalization(normalized);

  contextWidth = model->get_order() - 1;
  // For each state, we store at most contextWidth word ids to the left and
  // to the right and a kSTAR separator. The last bit represents the actual
  // size of the state.
  int max_state_size = (2 * contextWidth + 1) * sizeof(int) + 1;
  FeatureFunction::SetStateSize(max_state_size);

  vocab = model->get_vocabulary();
  mapper = make_shared<CdecNPLMMapper>(vocab);
  stateConverter = make_shared<CdecStateConverter>(max_state_size - 1);
  ruleConverter = make_shared<CdecRuleConverter>(mapper, stateConverter);

  kSTART = vocab->lookup_word("<s>");
  kSTOP = vocab->lookup_word("</s>");
  kUNKNOWN = vocab->lookup_word("<unk>");
  kSTAR = vocab->lookup_word("<{STAR}>");
}

void CdecNPLMWrapper::savePersistentCache() {
  if (persistentCache && cacheFile.size()) {
    ofstream f(cacheFile);
    boost::archive::binary_oarchive oa(f);
    cerr << "Saving n-gram probability cache to " << cacheFile << endl;
    oa << cache;
    cerr << "Finished saving " << cache.size()
         << " n-gram probabilities..." << endl;
  }
}

void CdecNPLMWrapper::loadPersistentCache(int sentence_id) {
  if (persistentCache) {
    cacheFile = filename + "." + to_string(sentence_id) + ".cache.bin";
    if (boost::filesystem::exists(cacheFile)) {
      ifstream f(cacheFile);
      boost::archive::binary_iarchive ia(f);
      cerr << "Loading n-gram probability cache from " << cacheFile << endl;
      ia >> cache;
      cerr << "Finished loading " << cache.size()
           << " n-gram probabilities..." << endl;
    } else {
      cerr << "Cache file not found..." << endl;
    }
  }
}

void CdecNPLMWrapper::PrepareForInput(const SentenceMetadata& smeta) {
  savePersistentCache();
  cache.clear();
  loadPersistentCache(smeta.GetSentenceId());
}

void CdecNPLMWrapper::TraversalFeaturesImpl(
    const SentenceMetadata& smeta, const HG::Edge& edge,
    const vector<const void*>& prev_states, SparseVector<double>* features,
    SparseVector<double>* estimated_features, void* next_state) const {
  vector<int> symbols = ruleConverter->convertTargetSide(
      edge.rule_->e(), prev_states);

  LMFeatures exact_scores = scoreFullContexts(symbols);
  if (exact_scores.LMScore) {
    features->set_value(fid, exact_scores.LMScore);
  }
  if (exact_scores.OOVScore) {
    features->set_value(fidOOV, exact_scores.OOVScore);
  }

  constructNextState(symbols, next_state);
  symbols = stateConverter->getTerminals(next_state);

  LMFeatures estimated_scores = estimateScore(symbols);
  if (estimated_scores.LMScore) {
    estimated_features->set_value(fid, estimated_scores.LMScore);
  }
  if (estimated_scores.OOVScore) {
    estimated_features->set_value(fidOOV, estimated_scores.OOVScore);
  }
}

void CdecNPLMWrapper::FinalTraversalFeatures(
    const void* prev_state, SparseVector<double>* features) const {
  vector<int> symbols = stateConverter->getTerminals(prev_state);
  symbols.insert(symbols.begin(), kSTART);
  symbols.push_back(kSTOP);

  LMFeatures final_scores = estimateScore(symbols);
  if (final_scores.LMScore) {
    features->set_value(fid, final_scores.LMScore);
  }
  if (final_scores.OOVScore) {
    features->set_value(fidOOV, final_scores.OOVScore);
  }
}

LMFeatures CdecNPLMWrapper::scoreFullContexts(
    const vector<int>& symbols) const {
  // Returns the sum of the scores of all the sequences of symbols other
  // than kSTAR that has length of at least ngram_order, score.
  LMFeatures ret;
  int last_star = -1;
  for (size_t i = 0; i < symbols.size(); ++i) {
    if (symbols[i] == kSTAR) {
      last_star = i;
    } else if (i - last_star > contextWidth) {
      ret += scoreContext(symbols, i);
    }
  }

  return ret;
}

double CdecNPLMWrapper::getScore(const vector<int>& ngram) const {
  return model->getScore(ngram);
}

LMFeatures CdecNPLMWrapper::scoreContext(
    const vector<int>& symbols, int position) const {
  int word = symbols[position];

  // Push up to the last contextWidth words into the context vector.
  // Note that the most recent context word is first, so if we're
  // scoring the word "diplomatic" with a 4-gram context in the sentence
  // "Australia is one of the few countries with diplomatic relations..."
  // the context vector would be ["with", "countries", "few"].
  vector<int> ngram;
  for (int i = 1; i <= contextWidth && position - i >= 0; ++i) {
    assert(symbols[position - i] != kSTAR);
    ngram.push_back(symbols[position - i]);
  }

  // If we haven't filled the full context, then pad it.
  // If the context hits the <s>, then pad with more <s>'s.
  // Otherwise, if the context is short due to a kSTAR,
  // pad with UNKs.
  if (!ngram.empty() && ngram.back() == kSTART) {
    ngram.resize(contextWidth, kSTART);
  } else {
    ngram.resize(contextWidth, kUNKNOWN);
  }

  reverse(ngram.begin(), ngram.end());
  ngram.push_back(word);

  // Check the cache for this context.
  // If it's in there, use the saved values as score.
  // Otherwise, run the full model to get the score value.
  double score;
  if (persistentCache) {
    ++totalHits;
    pair<double, bool> ret = cache.get(ngram);
    if (ret.second) {
      ++cacheHits;
      score = ret.first;
    } else {
      score = getScore(ngram);
      cache.put(ngram, score);
    }
  } else {
    score = getScore(ngram);
  }

  // Return the score, along with the OOV indicator feature value
  return LMFeatures(score, word == kUNKNOWN);
}

void CdecNPLMWrapper::constructNextState(
    const vector<int>& symbols, void* state) const {
  vector<int> next_state;
  for (size_t i = 0; i < symbols.size() && i < contextWidth; ++i) {
    if (symbols[i] == kSTAR) {
      break;
    }
    next_state.push_back(symbols[i]);
  }

  if (next_state.size() < symbols.size()) {
    next_state.push_back(kSTAR);

    int last_star = -1;
    for (size_t i = 0; i < symbols.size(); ++i) {
      if (symbols[i] == kSTAR) {
        last_star = i;
      }
    }

    size_t i = max(last_star + 1, static_cast<int>(symbols.size() - contextWidth));
    while (i < symbols.size()) {
      next_state.push_back(symbols[i]);
      ++i;
    }
  }

  stateConverter->convert(state, next_state);
}

LMFeatures CdecNPLMWrapper::estimateScore(const vector<int>& symbols) const {
  // Scores the symbols up to the first kSTAR, or up to the contextWidth,
  // whichever is first, padding the context with kSTART or kUNKNOWN as
  // needed. This offsets the fact that by scoreFullContexts() does not
  // score the first contextWidth words of a sentence.
  LMFeatures ret = scoreFullContexts(symbols);

  for (size_t i = 0; i < symbols.size() && i < contextWidth; ++i) {
    if (symbols[i] == kSTAR) {
      break;
    }

    if (symbols[i] != kSTART) {
      ret += scoreContext(symbols, i);
    }
  }

  return ret;
}

CdecNPLMWrapper::~CdecNPLMWrapper() {
  savePersistentCache();
  if (persistentCache) {
    cerr << "Cache hit ratio: " << 100.0 * cacheHits / totalHits
         << " %" << endl;
  }
}

} // namespace nplm
