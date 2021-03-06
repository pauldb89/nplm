#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "neuralClasses.h"
#include "Activation_function.h"

namespace nplm {

class model {
 public:
  Input_word_embeddings input_layer;
  Linear_layer first_hidden_linear;
  Activation_function first_hidden_activation;
  Linear_layer second_hidden_linear;
  Activation_function second_hidden_activation;
  Output_word_embeddings output_layer;

  Matrix<double,Dynamic,Dynamic,Eigen::RowMajor> output_embedding_matrix;
  Matrix<double,Dynamic,Dynamic,Eigen::RowMajor> input_embedding_matrix;
  Matrix<double,Dynamic,Dynamic,Eigen::RowMajor> input_and_output_embedding_matrix;

  activation_function_type activation_function;
  int ngram_size, vocab_size;
  int input_embedding_dimension, num_hidden, output_embedding_dimension;
  bool premultiplied;

  model(int ngram_size,
        int vocab_size,
        int input_embedding_dimension,
        int num_hidden,
        int output_embedding_dimension) {
    resize(
        ngram_size, vocab_size, input_embedding_dimension,
        num_hidden, output_embedding_dimension);
  }

  model()
      : ngram_size(1),
        premultiplied(false),
        activation_function(Rectifier),
        output_embedding_matrix(Matrix<double,Dynamic,Dynamic,Eigen::RowMajor>()),
        input_embedding_matrix(Matrix<double,Dynamic,Dynamic,Eigen::RowMajor>()) {}

  void resize(
      int ngram_size,
      int vocab_size,
      int input_embedding_dimension,
      int num_hidden,
      int output_embedding_dimension);

  void initialize(
      mt19937 &init_engine,
      bool init_normal,
      double init_range,
      double init_bias);

  void set_activation_function(activation_function_type f) {
    activation_function = f;
    first_hidden_activation.set_activation_function(f);
    second_hidden_activation.set_activation_function(f);
  }

  void premultiply();

  // Since the vocabulary is not essential to the model,
  // we need a version with and without a vocabulary.
  // If the number of "extra" data structures like this grows,
  // a better solution is needed

  void read(ifstream& filename);

  void write(ofstream& filename) const;

 private:
  void readConfig(ifstream& config_file);
};

} //namespace nplm

#endif
