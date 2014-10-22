#include "vocabulary.h"

#include <fstream>

namespace nplm {

Vocabulary::Vocabulary() {
  insert_word("<s>");
  insert_word("</s>");

  unk = insert_word("<unk>");

  insert_word("<null>");
}

int Vocabulary::lookup_word(const string& word) const {
  auto pos = index.find(word);
  return pos != index.end() ? pos->second : unk;
}

int Vocabulary::insert_word(const string& word) {
  auto it = index.insert(make_pair(word, words.size()));
  if (it.second) {
    words.push_back(word);
    return words.size();
  } else {
    return it.first->second;
  }
}

int Vocabulary::size() const {
  return words.size();
}

void Vocabulary::read(const string& filename) {
  ifstream fin(filename);
  string word;
  while (getline(fin, word)) {
    index[word] = words.size();
    words.push_back(word);
  }
}

void Vocabulary::write(const string& filename) const {
  ofstream fout(filename);
  for (const string& word: words) {
    fout << word << "\n";
  }
}

} // namespace nplm
