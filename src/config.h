#pragma once

#include <string>

using namespace std;

namespace nplm {

struct Config {
  string train_file;
  string validation_file;
  string test_file;

  string model_input_file;
  string model_output_file;

  string unigram_probs_file;
  string words_file;
  string input_words_file;
  string output_words_file;
  string model_prefix;

  int ngram_size;
  int vocab_size;
  int input_vocab_size;
  int output_vocab_size;
  int num_hidden;
  int embedding_dimension;
  int input_embedding_dimension;
  int output_embedding_dimension;
  string activation_function;
  string loss_function;

  int minibatch_size;
  int validation_minibatch_size;
  int num_epochs;
  double learning_rate;

  bool init_normal;
  double init_range;

  int num_noise_samples;

  bool use_momentum;
  double initial_momentum;
  double final_momentum;

  double L2_reg;

  bool normalization;
  double normalization_init;

  int num_threads;
};

} // namespace nplm
