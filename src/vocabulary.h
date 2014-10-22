#ifndef VOCABULARY_H
#define VOCABULARY_H

#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

namespace nplm {

class Vocabulary {
 public:
  Vocabulary();

  int lookup_word(const string &word) const;

  int insert_word(const string &word);

  int size() const;

  void read(const string& filename);

  void write(const string& filename) const;

 private:
  vector<string> words;
  unordered_map<string, int> index;
  int unk;
};

} // namespace nplm

#endif
