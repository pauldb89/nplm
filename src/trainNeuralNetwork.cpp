#include <algorithm>
#include <ctime>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include <boost/algorithm/string/join.hpp>
#include <boost/functional.hpp>
#include <boost/lexical_cast.hpp>

#include "../3rdparty/Eigen/Dense"
#include "../3rdparty/Eigen/Sparse"
#include "maybe_omp.h"
#include <tclap/CmdLine.h>

#include "context_extractor.h"
#include "corpus_utils.h"
#include "graphClasses.h"
#include "model.h"
#include "multinomial.h"
#include "neuralClasses.h"
#include "propagator.h"
#include "config.h"
#include "util.h"
#include "vocabulary.h"

//#define EIGEN_DONT_PARALLELIZE

using namespace std;
using namespace TCLAP;
using namespace Eigen;

using namespace nplm;

typedef unordered_map<VectorInt, double> VectorMap;

void EvaluateModel(
    const Config& config, const model& nn, propagator& prop_validation,
    const shared_ptr<Corpus>& test_corpus, const shared_ptr<Vocabulary>& vocab,
    int epoch, double& current_learning_rate, double& current_validation_ll) {
  if (test_corpus->size() > 0) {
    double log_likelihood = 0.0;
    Matrix<double,Dynamic,Dynamic> scores(vocab->size(), config.minibatch_size);
    Matrix<double,Dynamic,Dynamic> output_probs(vocab->size(), config.minibatch_size);

    cerr << endl;
    cerr << "Validation minibatches: " << endl;
    int num_batches = (test_corpus->size() - 1) / config.minibatch_size + 1;
    for (int batch = 0; batch < num_batches; batch++) {
      if (batch % 50 == 0) {
        cerr << batch << "... ";
      }

      data_size_t start_index = config.minibatch_size * batch;
      MatrixInt minibatch =
          ExtractMinibatch(test_corpus, vocab, config, start_index);

      prop_validation.fProp(minibatch.topRows(config.ngram_size - 1));

      // Do full forward prop through output word embedding layer
      start_timer(4);
      prop_validation.output_layer_node.param->fProp(prop_validation.second_hidden_activation_node.fProp_matrix, scores);
      stop_timer(4);

      // And softmax and loss. Be careful of short minibatch
      double minibatch_log_likelihood;
      start_timer(5);
      SoftmaxLogLoss().fProp(
          scores.leftCols(minibatch.cols()),
          minibatch.row(config.ngram_size - 1),
          output_probs,
          minibatch_log_likelihood);
      stop_timer(5);
      log_likelihood += minibatch_log_likelihood;
    }

    cerr << endl;
    cerr << "Validation log-likelihood: " << log_likelihood << endl;
    cerr << "           perplexity:     " << exp(-log_likelihood / test_corpus->size()) << endl;

    // If the validation perplexity decreases, halve the learning rate.
    if (epoch > 0 && log_likelihood < current_validation_ll) {
      current_learning_rate /= 2;
    } else {
      current_validation_ll = log_likelihood;

      if (config.model_output_file != "") {
        cerr << "Writing model to " << config.model_output_file << endl;
        ofstream fout(config.model_output_file);
        nn.write(fout);
        vocab->write(fout);
        cerr << "Done writing model" << endl;
      }
    }
  }
}

int main(int argc, char** argv) {
  Config config;
  try {
    // program options //
    CmdLine cmd("Trains a two-layer neural probabilistic language model.", ' ' , "0.1");

    // The options are printed in reverse order

    ValueArg<string> unigram_probs_file("", "unigram_probs_file", "Unigram model (deprecated and ignored)." , false, "", "string", cmd);

    ValueArg<int> num_threads("", "num_threads", "Number of threads. Default: maximum.", false, 0, "int", cmd);

    ValueArg<double> final_momentum("", "final_momentum", "Final value of momentum. Default: 0.9.", false, 0.9, "double", cmd);
    ValueArg<double> initial_momentum("", "initial_momentum", "Initial value of momentum. Default: 0.9.", false, 0.9, "double", cmd);
    ValueArg<bool> use_momentum("", "use_momentum", "Use momentum (hidden layer weights only). 1 = yes, 0 = no. Default: 0.", false, 0, "bool", cmd);

    ValueArg<double> normalization_init("", "normalization_init", "Initial normalization parameter. Default: 0.", false, 0.0, "double", cmd);
    ValueArg<bool> normalization("", "normalization", "Learn individual normalization factors during training. 1 = yes, 0 = no. Default: 0.", false, 0, "bool", cmd);

    ValueArg<int> num_noise_samples("", "num_noise_samples", "Number of noise samples for noise-contrastive estimation. Default: 25.", false, 25, "int", cmd);

    ValueArg<double> L2_reg("", "L2_reg", "L2 regularization strength (hidden layer weights only). Default: 0.", false, 0.0, "double", cmd);

    ValueArg<double> learning_rate("", "learning_rate", "Learning rate for stochastic gradient ascent. Default: 0.01.", false, 0.01, "double", cmd);

    ValueArg<int> minibatch_size("", "minibatch_size", "Minibatch size (for training). Default: 64.", false, 64, "int", cmd);

    ValueArg<int> num_epochs("", "num_epochs", "Number of epochs. Default: 10.", false, 10, "int", cmd);

    ValueArg<double> init_range("", "init_range", "Maximum (of uniform) or standard deviation (of normal) for initialization. Default: 0.01", false, 0.01, "double", cmd);
    ValueArg<bool> init_normal("", "init_normal", "Initialize parameters from a normal distribution. 1 = normal, 0 = uniform. Default: 0.", false, 0, "bool", cmd);

    ValueArg<string> loss_function("", "loss_function", "Loss function (log, nce). Default: nce.", false, "nce", "string", cmd);
    ValueArg<string> activation_function("", "activation_function", "Activation function (identity, rectifier, tanh, hardtanh, sigmoid). Default: rectifier.", false, "rectifier", "string", cmd);
    ValueArg<int> num_hidden("", "num_hidden", "Number of hidden nodes. Default: 100.", false, 100, "int", cmd);

    ValueArg<int> output_embedding_dimension("", "output_embedding_dimension", "Number of output embedding dimensions. Default: 50.", false, 50, "int", cmd);
    ValueArg<int> input_embedding_dimension("", "input_embedding_dimension", "Number of input embedding dimensions. Default: 50.", false, 50, "int", cmd);
    ValueArg<int> embedding_dimension("", "embedding_dimension", "Number of input and output embedding dimensions. Default: none.", false, -1, "int", cmd);

    ValueArg<int> ngram_size("", "ngram_size", "Size of n-grams. Default: auto.", false, 0, "int", cmd);

    ValueArg<string> model_output_file("", "model_output_file", "Model output file" , false, "", "string", cmd);
    ValueArg<string> validation_file("", "validation_file", "Validation data (one numberized example per line)." , false, "", "string", cmd);
    ValueArg<string> train_file("", "train_file", "Training data (one numberized example per line)." , true, "", "string", cmd);

    cmd.parse(argc, argv);

    // define program parameters //
    config.train_file = train_file.getValue();
    config.validation_file = validation_file.getValue();

    config.model_output_file = model_output_file.getValue();

    config.ngram_size = ngram_size.getValue();

    config.num_hidden = num_hidden.getValue();
    config.activation_function = activation_function.getValue();
    config.loss_function = loss_function.getValue();

    config.num_threads = num_threads.getValue();

    config.num_noise_samples = num_noise_samples.getValue();

    config.input_embedding_dimension = input_embedding_dimension.getValue();
    config.output_embedding_dimension = output_embedding_dimension.getValue();
    if (embedding_dimension.getValue() >= 0) {
      config.input_embedding_dimension = config.output_embedding_dimension = embedding_dimension.getValue();
    }

    config.minibatch_size = minibatch_size.getValue();
    config.num_epochs= num_epochs.getValue();
    config.learning_rate = learning_rate.getValue();
    config.use_momentum = use_momentum.getValue();
    config.normalization = normalization.getValue();
    config.initial_momentum = initial_momentum.getValue();
    config.final_momentum = final_momentum.getValue();
    config.L2_reg = L2_reg.getValue();
    config.init_normal= init_normal.getValue();
    config.init_range = init_range.getValue();
    config.normalization_init = normalization_init.getValue();

    cerr << "Command line: " << endl;
    cerr << boost::algorithm::join(vector<string>(argv, argv+argc), " ") << endl;

    const string sep(" Value: ");
    cerr << train_file.getDescription() << sep << train_file.getValue() << endl;
    cerr << validation_file.getDescription() << sep << validation_file.getValue() << endl;
    cerr << model_output_file.getDescription() << sep << model_output_file.getValue() << endl;

    cerr << ngram_size.getDescription() << sep << ngram_size.getValue() << endl;

    if (embedding_dimension.getValue() >= 0) {
      cerr << embedding_dimension.getDescription() << sep << embedding_dimension.getValue() << endl;
    } else {
      cerr << input_embedding_dimension.getDescription() << sep << input_embedding_dimension.getValue() << endl;
      cerr << output_embedding_dimension.getDescription() << sep << output_embedding_dimension.getValue() << endl;
    }
    cerr << num_hidden.getDescription() << sep << num_hidden.getValue() << endl;

    if (string_to_activation_function(activation_function.getValue()) == InvalidFunction) {
      cerr << "error: invalid activation function: " << activation_function.getValue() << endl;
      exit(1);
    }
    cerr << activation_function.getDescription() << sep << activation_function.getValue() << endl;

    if (string_to_loss_function(loss_function.getValue()) == InvalidLoss) {
      cerr << "error: invalid loss function: " << loss_function.getValue() << endl;
      exit(1);
    }
    cerr << loss_function.getDescription() << sep << loss_function.getValue() << endl;

    cerr << init_normal.getDescription() << sep << init_normal.getValue() << endl;
    cerr << init_range.getDescription() << sep << init_range.getValue() << endl;

    cerr << num_epochs.getDescription() << sep << num_epochs.getValue() << endl;
    cerr << minibatch_size.getDescription() << sep << minibatch_size.getValue() << endl;
    cerr << learning_rate.getDescription() << sep << learning_rate.getValue() << endl;
    cerr << L2_reg.getDescription() << sep << L2_reg.getValue() << endl;

    cerr << num_noise_samples.getDescription() << sep << num_noise_samples.getValue() << endl;

    cerr << normalization.getDescription() << sep << normalization.getValue() << endl;
    if (config.normalization) {
      cerr << normalization_init.getDescription() << sep << normalization_init.getValue() << endl;
    }

    cerr << use_momentum.getDescription() << sep << use_momentum.getValue() << endl;
    if (config.use_momentum) {
      cerr << initial_momentum.getDescription() << sep << initial_momentum.getValue() << endl;
      cerr << final_momentum.getDescription() << sep << final_momentum.getValue() << endl;
    }

    cerr << num_threads.getDescription() << sep << num_threads.getValue() << endl;

    if (unigram_probs_file.getValue() != "") {
      cerr << "Note: --unigram_probs_file is deprecated and ignored." << endl;
    }
  } catch (TCLAP::ArgException &e) {
    cerr << "error: " << e.error() <<  " for arg " << e.argId() << endl;
    exit(1);
  }

  config.num_threads = setup_threads(config.num_threads);
  int save_threads;

  //unsigned seed = std::time(0);
  unsigned seed = 1234; //for testing only
  mt19937 rng(seed);

  /////////////////////////READING IN THE TRAINING AND VALIDATION DATA///////////////////
  /////////////////////////////////////////////////////////////////////////////////////

  // Read training data
  shared_ptr<Vocabulary> vocab = make_shared<Vocabulary>();
  shared_ptr<Corpus> training_corpus =
      readCorpus(config.train_file, vocab);
  cerr << "Number of training instances: " << training_corpus->size() << endl;

  ContextExtractor extractor(
      training_corpus, config.ngram_size, vocab->lookup_word("<s>"),
      vocab->lookup_word("</s>"));

  // Shuffle training data for improved performance.
  random_shuffle(training_corpus->begin(), training_corpus->end());

  // Read validation data
  shared_ptr<Corpus> test_corpus;
  if (config.validation_file != "") {
    test_corpus = readCorpus(config.validation_file, vocab);
    cerr << "Number of validation instances: " << test_corpus->size() << endl;
  }

  ///// Construct unigram model and sampler that will be used for NCE
  vector<data_size_t> unigram_counts(vocab->size());
  for (data_size_t i = 0; i < training_corpus->size(); i++) {
    unigram_counts[training_corpus->at(i)] += 1;
  }
  multinomial<data_size_t> unigram(unigram_counts);

  ///// Create and initialize the neural network and associated propagators.

  model nn(
      config.ngram_size, vocab->size(),
      config.input_embedding_dimension, config.num_hidden,
      config.output_embedding_dimension);
  nn.initialize(rng, config.init_normal, config.init_range, -log(vocab->size()));
  nn.set_activation_function(string_to_activation_function(config.activation_function));

  loss_function_type loss_function = string_to_loss_function(config.loss_function);

  propagator prop(nn, config.minibatch_size);
  propagator prop_validation(nn, config.minibatch_size);
  SoftmaxNCELoss<multinomial<data_size_t> > softmax_loss(unigram);
  // normalization parameters
  VectorMap c_h, c_h_running_gradient;

  ///////////////////////TRAINING THE NEURAL NETWORK////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////

  data_size_t num_batches = (training_corpus->size() - 1) / config.minibatch_size + 1;
  cerr << "Number of training minibatches: " << num_batches << endl;

  int num_validation_batches = 0;
  if (test_corpus->size() > 0) {
    num_validation_batches = (test_corpus->size() - 1) / config.minibatch_size + 1;
    cerr << "Number of validation minibatches: "
         << num_validation_batches << endl;
  }

  double current_momentum = config.initial_momentum;
  double momentum_delta = (config.final_momentum - config.initial_momentum)/(config.num_epochs-1);
  double current_learning_rate = config.learning_rate;
  double current_validation_ll = 0.0;

  int ngram_size = config.ngram_size;
  data_size_t minibatch_size = config.minibatch_size;
  int num_noise_samples = config.num_noise_samples;

  if (config.normalization) {
    for (data_size_t i = 0; i < training_corpus->size(); i++) {
      VectorInt context = extractor.extract(i);
      if (c_h.find(context) == c_h.end()) {
        c_h[context] = -config.normalization_init;
      }
    }
  }

  for (int epoch = 0; epoch < config.num_epochs; epoch++) {
    cerr << "Epoch " << epoch + 1 << endl;
    cerr << "Current learning rate: " << current_learning_rate << endl;

    if (config.use_momentum) {
      cerr << "Current momentum: " << current_momentum << endl;
    } else {
      current_momentum = -1;
    }

    cerr << "Training minibatches: " << endl;

    double log_likelihood = 0.0;

    int num_samples = 0;
    if (loss_function == LogLoss) {
      num_samples = vocab->size();
    } else if (loss_function == NCELoss) {
      num_samples = 1 + num_noise_samples;
    }

    Matrix<double,Dynamic,Dynamic> minibatch_weights(num_samples, minibatch_size);
    Matrix<int,Dynamic,Dynamic> minibatch_samples(num_samples, minibatch_size);
    Matrix<double,Dynamic,Dynamic> scores(num_samples, minibatch_size);
    Matrix<double,Dynamic,Dynamic> probs(num_samples, minibatch_size);

    for (data_size_t batch = 0; batch < num_batches; batch++) {
      if (batch % 10000 == 0) {
        cerr << batch << "... ";
        if (batch % 200000 == 0) {
          EvaluateModel(
              config, nn, prop_validation, test_corpus, vocab,
              epoch, current_learning_rate, current_validation_ll);
          cerr << "Current learning rate: " << current_learning_rate << endl;
          cerr << "Training minibatches: " << endl;
        }
      }

      data_size_t start_index = minibatch_size * batch;
      MatrixInt minibatch =
          ExtractMinibatch(training_corpus, vocab, config, start_index);

      double adjusted_learning_rate = current_learning_rate / minibatch.cols();
            ///// Forward propagation
      prop.fProp(minibatch.topRows(ngram_size-1));

      if (loss_function == NCELoss) {
        ///// Noise-contrastive estimation

        // Generate noise samples. Gather positive and negative samples into matrix.

        start_timer(3);

        minibatch_samples.block(0, 0, 1, minibatch.cols()) = minibatch.bottomRows(1);

        for (int sample_id = 1; sample_id < num_noise_samples+1; sample_id++) {
          for (int train_id = 0; train_id < minibatch.cols(); train_id++) {
            minibatch_samples(sample_id, train_id) = unigram.sample(rng);
          }
        }

        stop_timer(3);

        // Final forward propagation step (sparse)
        start_timer(4);
        prop.output_layer_node.param->fProp(
            prop.second_hidden_activation_node.fProp_matrix,
            minibatch_samples, scores);
        stop_timer(4);

        // Apply normalization parameters
        if (config.normalization)
        {
          for (int train_id = 0;train_id < minibatch.cols(); train_id++)
          {
            VectorInt context = minibatch.block(0, train_id, ngram_size-1, 1);
            scores.col(train_id).array() += c_h[context];
          }
        }

        double minibatch_log_likelihood;
        start_timer(5);
        // Compute conditional probabilities p(C | w, h) that a word was drawn
        // from the data or from the noise (unigram) distribution.
        softmax_loss.fProp(scores.leftCols(minibatch.cols()),
            minibatch_samples,
            probs, minibatch_log_likelihood);
        stop_timer(5);
        log_likelihood += minibatch_log_likelihood;

        ///// Backward propagation

        start_timer(6);
        softmax_loss.bProp(probs, minibatch_weights);
        stop_timer(6);

        // Update the normalization parameters
        if (config.normalization)
        {
          for (int train_id = 0;train_id < minibatch.cols(); train_id++)
          {
            Matrix<int,Dynamic,1> context = minibatch.block(0, train_id, ngram_size-1, 1);
            c_h[context] += adjusted_learning_rate * minibatch_weights.col(train_id).sum();
          }
        }

        // Be careful of short minibatch
        prop.bProp(minibatch.topRows(ngram_size-1),
            minibatch_samples.leftCols(minibatch.cols()),
            minibatch_weights.leftCols(minibatch.cols()),
            adjusted_learning_rate, current_momentum, config.L2_reg);
      } else if (loss_function == LogLoss) {
        ///// Standard log-likelihood
        start_timer(4);
        prop.output_layer_node.param->fProp(prop.second_hidden_activation_node.fProp_matrix, scores);
        stop_timer(4);

        double minibatch_log_likelihood;
        start_timer(5);
        SoftmaxLogLoss().fProp(scores.leftCols(minibatch.cols()),
            minibatch.row(ngram_size-1),
            probs,
            minibatch_log_likelihood);
        stop_timer(5);
        log_likelihood += minibatch_log_likelihood;

        ///// Backward propagation

        start_timer(6);
        SoftmaxLogLoss().bProp(minibatch.row(ngram_size-1).leftCols(minibatch.cols()),
            probs.leftCols(minibatch.cols()),
            minibatch_weights);
        stop_timer(6);

        prop.bProp(minibatch.topRows(ngram_size-1).leftCols(minibatch.cols()),
            minibatch_weights,
            adjusted_learning_rate, current_momentum, config.L2_reg);
      }
    }

    cerr << "done." << endl;

    if (loss_function == LogLoss) {
      cerr << "Training log-likelihood: " << log_likelihood << endl;
      cerr << "         perplexity:     " << exp(-log_likelihood/training_corpus->size()) << endl;
    } else if (loss_function == NCELoss) {
      cerr << "Training NCE log-likelihood: " << log_likelihood << endl;
    }

    current_momentum += momentum_delta;

    #ifdef USE_CHRONO
    cerr << "Propagation times:";
    for (int i = 0; i < timer.size(); i++) {
      cerr << " " << timer.get(i);
    }
    cerr << endl;
    #endif

    EvaluateModel(
        config, nn, prop_validation, test_corpus, vocab,
        epoch, current_learning_rate, current_validation_ll);
  }

  return 0;
}
