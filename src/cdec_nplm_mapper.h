#pragma once

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "vocabulary.h"

using namespace std;

namespace nplm {

class CdecNPLMMapper {
 public:
  CdecNPLMMapper(const shared_ptr<Vocabulary>& vocab);

  int convert(int cdec_id) const;

 private:
  void add(int nplm_id, int cdec_id);

  shared_ptr<Vocabulary> vocab;
  vector<int> cdec2nplm;
  int kUNKNOWN;
};

} // namespace nplm
