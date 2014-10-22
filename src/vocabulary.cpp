#include "vocabulary.h"

#include <fstream>
#include <iostream>
#include <sstream>

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
  auto ret = index.insert(make_pair(word, words.size()));
  if (ret.second) {
    words.push_back(word);
  }
  return ret.first->second;
}

int Vocabulary::size() const {
  return words.size();
}

void Vocabulary::read(ifstream& fin) {
  string line;
  while (getline(fin, line)) {
    string word;
    stringstream ss(line);
    while (ss >> word) {
      insert_word(word);
    }
  }
}

void Vocabulary::write(ofstream& fout) const {
  for (const string& word: words) {
    fout << word << "\n";
  }
}

} // namespace nplm
