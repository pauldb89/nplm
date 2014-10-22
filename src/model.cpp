#include <cstdlib>
#include <iostream>
#include <boost/lexical_cast.hpp>

#include "model.h"

using namespace std;

namespace nplm {

void model::resize(
    int ngram_size, int vocab_size,
    int input_embedding_dimension, int num_hidden,
    int output_embedding_dimension) {
  input_layer.resize(vocab_size, input_embedding_dimension, ngram_size-1);
  first_hidden_linear.resize(num_hidden, input_embedding_dimension * (ngram_size-1));
  first_hidden_activation.resize(num_hidden);
  second_hidden_linear.resize(output_embedding_dimension, num_hidden);
  second_hidden_activation.resize(output_embedding_dimension);
  output_layer.resize(vocab_size, output_embedding_dimension);
  this->ngram_size = ngram_size;
  this->vocab_size = vocab_size;
  this->input_embedding_dimension = input_embedding_dimension;
  this->num_hidden = num_hidden;
  this->output_embedding_dimension = output_embedding_dimension;
  premultiplied = false;
}

void model::initialize(
    mt19937 &init_engine,
    bool init_normal,
    double init_range,
    double init_bias) {
  input_layer.initialize(init_engine, init_normal, init_range);
  output_layer.initialize(init_engine, init_normal, init_range, init_bias);
  first_hidden_linear.initialize(init_engine, init_normal, init_range);
  second_hidden_linear.initialize(init_engine, init_normal, init_range);
}

void model::premultiply() {
  cerr << "Premultiplying NPLM" << endl;

  // Since input and first_hidden_linear are both linear,
  // we can multiply them into a single linear layer *if* we are not training
  int context_size = ngram_size - 1;
  Matrix<double, Dynamic, Dynamic> U = first_hidden_linear.U;
  first_hidden_linear.U.resize(num_hidden, vocab_size * context_size);
  for (int i = 0; i < context_size; i++) {
      first_hidden_linear.U.middleCols(i * vocab_size, vocab_size) =
          U.middleCols(i * input_embedding_dimension, input_embedding_dimension) * input_layer.W.transpose();
  }
  input_layer.W.resize(1, 1); // try to save some memory
  premultiplied = true;
}

void model::readConfig(ifstream &fin) {
  string line;
  vector<string> fields;
  int ngram_size, vocab_size, input_embedding_dimension, num_hidden, output_embedding_dimension;
  activation_function_type activation_function = this->activation_function;
  while (getline(fin, line) && line != "") {
    splitBySpace(line, fields);
	  if (fields[0] == "ngram_size") {
	    ngram_size = boost::lexical_cast<int>(fields[1]);
    } else if (fields[0] == "vocab_size") {
      vocab_size = boost::lexical_cast<int>(fields[1]);
    }else if (fields[0] == "input_embedding_dimension") {
	    input_embedding_dimension = boost::lexical_cast<int>(fields[1]);
    } else if (fields[0] == "num_hidden") {
	    num_hidden = boost::lexical_cast<int>(fields[1]);
    } else if (fields[0] == "output_embedding_dimension") {
	    output_embedding_dimension = boost::lexical_cast<int>(fields[1]);
    }	else if (fields[0] == "activation_function") {
	    activation_function = string_to_activation_function(fields[1]);
	  } else if (fields[0] == "version") {
	    int version = boost::lexical_cast<int>(fields[1]);
	    if (version != 1) {
    		cerr << "error: file format mismatch (expected 1, found " << version << ")" << endl;
		    exit(1);
	    }
	  } else {
	    cerr << "warning: unrecognized field in config: " << fields[0] << endl;
    }
  }

  resize(
      ngram_size, vocab_size,
      input_embedding_dimension, num_hidden, output_embedding_dimension);
  set_activation_function(activation_function);
}

void model::read(ifstream& fin) {
  if (!fin) {
    throw runtime_error("Could not open file");
  }

  string line;

  while (getline(fin, line)) {
  	if (line == "\\config") {
	    readConfig(fin);
	  } else if (line == "\\input_embeddings") {
	    input_layer.read(fin);
    } else if (line == "\\hidden_weights 1") {
	    first_hidden_linear.read(fin);
    } else if (line == "\\hidden_weights 2") {
	    second_hidden_linear.read(fin);
    } else if (line == "\\output_weights") {
	    output_layer.read_weights(fin);
    } else if (line == "\\output_biases") {
	    output_layer.read_biases(fin);
    } else if (line == "\\end") {
	    break;
    } else if (line == "") {
	    continue;
    } else {
	    cerr << "warning: unrecognized section: " << line << endl;
	    // skip over section
	    while (getline(fin, line) && line != "") { }
	  }
  }
}

void model::write(ofstream& fout) const {
  if (!fout) {
    throw runtime_error("Could not open file");
  }

  fout << "\\config" << endl;
  fout << "version 1" << endl;
  fout << "ngram_size " << ngram_size << endl;
  fout << "vocab_size " << vocab_size << endl;
  fout << "input_embedding_dimension " << input_embedding_dimension << endl;
  fout << "num_hidden " << num_hidden << endl;
  fout << "output_embedding_dimension " << output_embedding_dimension << endl;
  fout << "activation_function " << activation_function_to_string(activation_function) << endl;
  fout << endl;

  fout << "\\input_embeddings" << endl;
  input_layer.write(fout);
  fout << endl;

  fout << "\\hidden_weights 1" << endl;
  first_hidden_linear.write(fout);
  fout << endl;

  fout << "\\hidden_weights 2" << endl;
  second_hidden_linear.write(fout);
  fout << endl;

  fout << "\\output_weights" << endl;
  output_layer.write_weights(fout);
  fout << endl;

  fout << "\\output_biases" << endl;
  output_layer.write_biases(fout);
  fout << endl;

  fout << "\\end" << endl;
}

} // namespace nplm
