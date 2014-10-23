#include "corpus_utils.h"

#include "context_extractor.h"

namespace nplm {

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
