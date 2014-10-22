#include <tclap/CmdLine.h>
#include <boost/algorithm/string/join.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>

#include "model.h"
#include "propagator.h"
#include "neuralClasses.h"
#include "config.h"
#include "util.h"

using namespace std;
using namespace boost;
using namespace TCLAP;
using namespace Eigen;

using namespace nplm;

int main (int argc, char *argv[])
{
    Config config;

    try {
      // program options //
      CmdLine cmd("Tests a two-layer neural probabilistic language model.", ' ' , "0.1");

      ValueArg<int> num_threads("", "num_threads", "Number of threads. Default: maximum.", false, 0, "int", cmd);
      ValueArg<int> minibatch_size("", "minibatch_size", "Minibatch size. Default: 64.", false, 64, "int", cmd);

      ValueArg<string> arg_test_file("", "test_file", "Test file (one numberized example per line).", true, "", "string", cmd);

      ValueArg<string> arg_model_input_file("", "model_input_file", "Model file.", true, "", "string", cmd);

      cmd.parse(argc, argv);

      config.model_input_file = arg_model_input_file.getValue();
      config.test_file = arg_test_file.getValue();

      config.num_threads  = num_threads.getValue();
      config.minibatch_size = minibatch_size.getValue();

      cerr << "Command line: " << endl;
      cerr << boost::algorithm::join(vector<string>(argv, argv+argc), " ") << endl;

      const string sep(" Value: ");
      cerr << arg_model_input_file.getDescription() << sep << arg_model_input_file.getValue() << endl;
      cerr << arg_test_file.getDescription() << sep << arg_test_file.getValue() << endl;

      cerr << num_threads.getDescription() << sep << num_threads.getValue() << endl;
    }
    catch (TCLAP::ArgException &e)
    {
      cerr << "error: " << e.error() <<  " for arg " << e.argId() << endl;
      exit(1);
    }

    config.num_threads = setup_threads(config.num_threads);

    ///// Create network and propagator

    model nn;
    nn.read(config.model_input_file);
    config.ngram_size = nn.ngram_size;
    propagator prop(nn, config.minibatch_size);

    ///// Set param values according to what was read in from model file

    config.ngram_size = nn.ngram_size;
    config.num_hidden = nn.num_hidden;
    config.input_embedding_dimension = nn.input_embedding_dimension;
    config.output_embedding_dimension = nn.output_embedding_dimension;

    ///// Read test data

    vector<int> test_data_flat;
    readDataFile(config.test_file, config.ngram_size, test_data_flat);
    int test_data_size = test_data_flat.size() / config.ngram_size;
    cerr << "Number of test instances: " << test_data_size << endl;

    Map< Matrix<int,Dynamic,Dynamic> > test_data(test_data_flat.data(), config.ngram_size, test_data_size);

    ///// Score test data

    int num_batches = (test_data_size-1)/config.minibatch_size + 1;
    cerr<<"Number of test minibatches: "<<num_batches<<endl;

    double log_likelihood = 0.0;

    Matrix<double,Dynamic,Dynamic> scores(nn.vocab_size, config.minibatch_size);
    Matrix<double,Dynamic,Dynamic> output_probs(nn.vocab_size, config.minibatch_size);

    for (int batch = 0; batch < num_batches; batch++)
    {
  int minibatch_start_index = config.minibatch_size * batch;
  int current_minibatch_size = min(config.minibatch_size,
           test_data_size - minibatch_start_index);
  Matrix<int,Dynamic,Dynamic> minibatch = test_data.middleCols(minibatch_start_index, current_minibatch_size);

  prop.fProp(minibatch.topRows(config.ngram_size-1));

  // Do full forward prop through output word embedding layer
  prop.output_layer_node.param->fProp(prop.second_hidden_activation_node.fProp_matrix, scores);

  // And softmax and loss
  double minibatch_log_likelihood;
  SoftmaxLogLoss().fProp(scores.leftCols(current_minibatch_size),
             minibatch.row(config.ngram_size-1),
             output_probs,
             minibatch_log_likelihood);
  log_likelihood += minibatch_log_likelihood;

  /*for (int i=0; i<current_minibatch_size; i++)
    cerr << minibatch.block(0,i,config.ngram_size,1) << " " << output_probs(minibatch(config.ngram_size-1,i),i) << endl;*/

    }
    cerr << "Test log-likelihood: " << log_likelihood << endl;
}
