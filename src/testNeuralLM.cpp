#include <algorithm>
#include <fstream>
#include <memory>

#include <boost/algorithm/string/join.hpp>
#include <tclap/CmdLine.h>

#include "../3rdparty/Eigen/Core"
#include "../3rdparty/Eigen/Dense"

#include "config.h"
#include "corpus_utils.h"
#include "minibatch_extractor.h"
#include "neuralLM.h"

using namespace std;
using namespace TCLAP;
using namespace Eigen;

using namespace nplm;

int main(int argc, char *argv[]) {
  Config config;

  try {
    // program options //
    CmdLine cmd("Tests a two-layer neural probabilistic language model.", ' ' , "0.1");

    ValueArg<int> num_threads("", "num_threads", "Number of threads. Default: maximum.", false, 0, "int", cmd);
    ValueArg<int> minibatch_size("", "minibatch_size", "Minibatch size. Default: none.", false, 0, "int", cmd);
    ValueArg<string> arg_test_file("", "test_file", "Test file (one tokenized sentence per line).", true, "", "string", cmd);
    ValueArg<string> arg_model_input_file("", "model_input_file", "Language model file.", true, "", "string", cmd);
    ValueArg<bool> arg_normalization("", "normalization", "Normalize probabilities. 1 = yes, 0 = no. Default: 1.", false, 1, "bool", cmd);

    cmd.parse(argc, argv);

    config.model_input_file = arg_model_input_file.getValue();
    config.test_file = arg_test_file.getValue();
    config.minibatch_size = minibatch_size.getValue();
    config.num_threads = num_threads.getValue();
    config.normalization = arg_normalization.getValue();

    cerr << "Command line: " << endl;
    cerr << boost::algorithm::join(vector<string>(argv, argv+argc), " ") << endl;

    const string sep(" Value: ");
    cerr << arg_test_file.getDescription() << sep << arg_test_file.getValue() << endl;
    cerr << arg_model_input_file.getDescription() << sep << arg_model_input_file.getValue() << endl;

    cerr << minibatch_size.getDescription() << sep << minibatch_size.getValue() << endl;
    cerr << num_threads.getDescription() << sep << num_threads.getValue() << endl;
  } catch (TCLAP::ArgException &e) {
    cerr << "error: " << e.error() <<  " for arg " << e.argId() << endl;
    exit(1);
  }

  config.num_threads = setup_threads(config.num_threads);

  ///// Create language model

  neuralLM lm(config.model_input_file);
  lm.set_normalization(config.normalization);
  lm.set_cache(1048576);
  config.ngram_size = lm.get_order();
  size_t minibatch_size = config.minibatch_size;
  lm.set_width(minibatch_size);

  shared_ptr<Corpus> test_corpus =
      readCorpus(config.test_file, lm.get_vocabulary());

  MinibatchExtractor extractor(test_corpus, lm.get_vocabulary(), config);
  double log_likelihood = 0;
  for (int test_id = 0; test_id < test_corpus->size(); test_id += minibatch_size) {
    MatrixInt minibatch = extractor.extract(test_id);
    Matrix<double, 1, Dynamic> log_probs(minibatch.cols());
    lm.lookup_ngram(minibatch.leftCols(minibatch.cols()), log_probs);
    log_likelihood += log_probs.sum();
  }

  cerr << "Test log10-likelihood: " << log_likelihood << endl;
  cerr << "Perplexity: " << exp(-log_likelihood / test_corpus->size()) << endl;
#ifdef USE_CHRONO
  cerr << "Propagation times:";
  for (int i = 0; i < timer.size(); i++) {
    cerr << " " << timer.get(i);
  }
  cerr << endl;
#endif

  return 0;
}
