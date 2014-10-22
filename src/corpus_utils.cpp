#include "corpus_utils.h"

#include "context_extractor.h"

namespace nplm {

MatrixInt ExtractMinibatch(
    const shared_ptr<Corpus>& corpus,
    const shared_ptr<Vocabulary>& vocab,
    const Config& config,
    data_size_t start_index) {
  data_size_t actual_size = min(
      static_cast<data_size_t>(corpus->size()) - start_index,
      static_cast<data_size_t>(config.minibatch_size));
  ContextExtractor extractor(
      corpus, config.ngram_size, vocab->lookup_word("<s>"),
      vocab->lookup_word("</s>"));
  MatrixInt minibatch(config.ngram_size, actual_size);
  for (data_size_t i = 0; i < actual_size; ++i) {
    minibatch.col(i) = extractor.extract(start_index + i);
  }

  return minibatch;
}

shared_ptr<Corpus> readCorpus(
    const string& filename,
    const shared_ptr<Vocabulary>& vocab) {
  shared_ptr<Corpus> corpus = make_shared<Corpus>();

  int eos_id = vocab->lookup_word("</s>");

  string line;
  ifstream fin(filename);
  int line_id = 0;
  while (getline(fin, line)) {
    ++line_id;
    if (line_id % 100000 == 0) {
      cerr << line_id << "...";
    }
    string word;
    stringstream stream(line);
    while (stream >> word) {
      corpus->push_back(vocab->insert_word(word));
    }

    corpus->push_back(eos_id);
  }
  cerr << endl;

  return corpus;
}

} // namespace nplm
