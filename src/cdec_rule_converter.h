#pragma once

#include <vector>

#include <boost/shared_ptr.hpp>

#include "cdec_nplm_mapper.h"
#include "cdec_state_converter.h"

using namespace std;

namespace nplm {

class CdecRuleConverter {
 public:
  CdecRuleConverter(
      const shared_ptr<CdecNPLMMapper>& mapper,
      const shared_ptr<CdecStateConverterBase>& state_converter);

  vector<int> convertTargetSide(
      const vector<int>& target, const vector<const void*>& prev_states) const;

 private:
  shared_ptr<CdecNPLMMapper> mapper;
  shared_ptr<CdecStateConverterBase> stateConverter;
};

} // namespace nplm
