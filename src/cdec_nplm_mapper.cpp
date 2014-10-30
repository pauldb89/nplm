#include "cdec_nplm_mapper.h"

#include "hg.h"

namespace nplm {

CdecNPLMMapper::CdecNPLMMapper(
    const shared_ptr<Vocabulary>& vocab) : vocab(vocab) {
  kUNKNOWN = this->vocab->lookup_word("<unk>");
  for (int i = 0; i < vocab->size(); ++i) {
    add(i, TD::Convert(vocab->lookup_id(i)));
  }
}

void CdecNPLMMapper::add(int nplm_id, int cdec_id) {
  if (cdec_id >= cdec2nplm.size()) {
    cdec2nplm.resize(cdec_id + 1, kUNKNOWN);
  }
  cdec2nplm[cdec_id] = nplm_id;
}

int CdecNPLMMapper::convert(int cdec_id) const {
  if (cdec_id < 0 || cdec_id >= cdec2nplm.size()) {
    return kUNKNOWN;
  } else {
    return cdec2nplm[cdec_id];
  }
}

} // namespace nplm
