#ifndef NEURALLM_H
#define NEURALLM_H

#include <chrono>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <cctype>
#include <cstdlib>

#include <boost/functional/hash.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>

#include "../3rdparty/Eigen/Dense"

#include "util.h"
#include "model.h"
#include "propagator.h"
#include "neuralClasses.h"
#include "vocabulary.h"

#ifdef WITH_THREADS // included in multi-threaded moses project
#endif

using namespace std;

namespace nplm {

class NeuralModel {
 public:
  explicit NeuralModel(const string& filename, bool premultiply = false) {
    ifstream fin(filename);
    nn = make_shared<model>();
    nn->read(fin);

    vocab = make_shared<Vocabulary>();
    vocab->read(fin);

    if (premultiply) {
      nn->premultiply();
    }
  }

  int getOrder() const {
    return nn->ngram_size;
  }

  shared_ptr<model> getModel() const {
    return nn;
  }

  shared_ptr<Vocabulary> getVocabulary() const {
    return vocab;
  }

 private:
  shared_ptr<model> nn;
  shared_ptr<Vocabulary> vocab;
};

class NeuralLMCache {
 public:
  double lookup(const vector<int>& ngram) const {
    auto it = cache.find(ngram);
    return it == cache.end() ? 0 : it->second;
  }

  void store(const vector<int>& ngram, double value) {
    cache[ngram] = value;
  }

  void clear() {
    cache.clear();
  }

  void read(const string& filename) {
    ifstream fin(filename);
    size_t key_size;
    while (fin >> key_size) {
      double value;
      vector<int> key(key_size);
      for (size_t i = 0; i < key_size; ++i) {
        fin >> key[i];
      }
      fin >> value;
      cache[key] = value;
    }
  }

  void write(const string& filename) const {
    ofstream fout(filename);
    for (const auto& entry: cache) {
      fout << entry.first.size() << " ";
      for (int word_id: entry.first) {
        fout << word_id << " ";
      }
      fout << entry.second << "\n";
    }
  }

 private:
  unordered_map<vector<int>, double, boost::hash<vector<int>>> cache;
};

class NeuralLM {
  shared_ptr<NeuralModel> sharedModel;
  NeuralLMCache cache;
  shared_ptr<Vocabulary> vocab;

  bool normalization;
  char map_digits;

  propagator prop;

  int ngram_size;
  int width;

  double weight;
  int START_ID, NULL_ID;

 public:
  NeuralLM(const string &filename, bool premultiply = false)
      : normalization(false), weight(1.), map_digits(0), width(1) {
    sharedModel = make_shared<NeuralModel>(filename, premultiply);

    ngram_size = sharedModel->getOrder();
    vocab = sharedModel->getVocabulary();

    prop = propagator(*sharedModel->getModel(), 1);
    prop.resize();

    START_ID = vocab->lookup_word("<s>");
    NULL_ID = vocab->lookup_word("<null>");
  }

  // initialize NeuralLM class that shares vocab and model with base instance (for multithreaded decoding)
  NeuralLM(const NeuralLM& base)
    : sharedModel(base.sharedModel),
      ngram_size(base.ngram_size),
      normalization(base.normalization),
      weight(base.weight),
      map_digits(base.map_digits),
      width(base.width),
      prop(*sharedModel->getModel(), 1),
      vocab(sharedModel->getVocabulary()) {
    prop.resize();

    START_ID = vocab->lookup_word("<s>");
    NULL_ID = vocab->lookup_word("<null>");
  }

  void set_normalization(bool value) {
    normalization = value;
  }

  void set_log_base(double value) {
    weight = 1 / log(value);
  }

  void set_map_digits(char value) {
    map_digits = value;
  }

  void set_width(int width) {
    this->width = width;
    prop.resize(width);
  }

  shared_ptr<Vocabulary> get_vocabulary() const {
    return vocab;
  }

  Vocabulary get_vocab() const {
    return *vocab;
  }

  int lookup_input_word(const string &word) const {
    if (!map_digits) {
      return vocab->lookup_word(word);
    }

    string mapped_word(word);
    for (int i = 0; i < word.length(); i++) {
      if (isdigit(word[i])) {
        mapped_word[i] = map_digits;
      }
    }

    return vocab->lookup_word(mapped_word);
  }

  int lookup_word(const string &word) const {
    return lookup_input_word(word);
  }

  int lookup_output_word(const string &word) const {
    if (!map_digits) {
      return vocab->lookup_word(word);
    }

    string mapped_word(word);
    for (int i = 0; i < word.length(); ++i) {
      if (isdigit(word[i])) {
        mapped_word[i] = map_digits;
      }
    }

    return vocab->lookup_word(mapped_word);
  }

  void clear_cache() {
    cache.clear();
  }

  void load_cache(const string& filename) {
    cache.read(filename);
  }

  void save_cache(const string& filename) const {
    cache.write(filename);
  }

  double score_ngram(const vector<int>& query) {
    assert(query.size() == ngram_size);

    double cached = cache.lookup(query);
    if (cached != 0) {
      return cached;
    }

    // Make sure that we're single threaded. Multithreading doesn't help,
    // and in some cases can hurt quite a lot
    int save_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    int save_eigen_threads = Eigen::nbThreads();
    Eigen::setNbThreads(1);
#ifdef __INTEL_MKL__
    int save_mkl_threads = mkl_get_max_threads();
    mkl_set_num_threads(1);
#endif

    VectorInt ngram(query.size());
    for (size_t i = 0; i < query.size(); ++i) {
      ngram(i) = query[i];
    }

    prop.fProp(ngram);

    int output = ngram(ngram_size - 1, 0);
    double log_prob;

    start_timer(3);
    if (normalization) {
      Eigen::Matrix<double,Eigen::Dynamic,1> scores(vocab->size());
      prop.output_layer_node.param->fProp(prop.second_hidden_activation_node.fProp_matrix, scores);
      double logz = logsum(scores.col(0));
      log_prob = weight * (scores(output, 0) - logz);
    } else {
      log_prob = weight * prop.output_layer_node.param->fProp(
          prop.second_hidden_activation_node.fProp_matrix, output, 0);
    }
    stop_timer(3);

    cache.store(query, log_prob);

#ifndef WITH_THREADS
#ifdef __INTEL_MKL__
    mkl_set_num_threads(save_mkl_threads);
#endif
    Eigen::setNbThreads(save_eigen_threads);
    omp_set_num_threads(save_threads);
#endif
    return log_prob;
  }

  // Look up many n-grams in parallel.
  template <typename DerivedA, typename DerivedB>
  void lookup_ngram(
      const Eigen::MatrixBase<DerivedA> &ngram,
      const Eigen::MatrixBase<DerivedB> &log_probs_const) {
    UNCONST(DerivedB, log_probs_const, log_probs);
    assert(ngram.rows() == ngram_size);
    assert(ngram.cols() <= width);

    prop.fProp(ngram);

    if (normalization) {
      Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> scores(
          vocab->size(), ngram.cols());
      prop.output_layer_node.param->fProp(
          prop.second_hidden_activation_node.fProp_matrix, scores);

      // And softmax and loss
      Matrix<double,Dynamic,Dynamic> output_probs(
          vocab->size(), ngram.cols());
      double minibatch_log_likelihood;
      SoftmaxLogLoss().fProp(
          scores.leftCols(ngram.cols()), ngram.row(ngram_size - 1),
          output_probs, minibatch_log_likelihood);

      for (int j = 0; j < ngram.cols(); j++) {
        int output = ngram(ngram_size - 1, j);
        log_probs(0, j) = weight * output_probs(output, j);
      }
    } else {
      for (int j = 0; j < ngram.cols(); j++) {
        int output = ngram(ngram_size - 1, j);
        log_probs(0, j) = weight * prop.output_layer_node.param->fProp(
            prop.second_hidden_activation_node.fProp_matrix, output, j);
      }
    }
  }

  double lookup_ngram(const vector<int>& query) {
    auto start_time = GetTime();
    vector<int> ngram;
    for (int i = 0; i < ngram_size; ++i) {
      if (i < ngram_size - query.size()) {
        ngram.push_back(query[0] == START_ID ? START_ID : NULL_ID);
      } else {
        ngram.push_back(query[i - ngram_size + query.size()]);
      }
    }

    return score_ngram(ngram);
  }

  int get_order() const {
    return ngram_size;
  }
};

} // namespace nplm

#endif
