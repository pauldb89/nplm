#pragma once

#include <memory>
#include <string>

#include "config.h"
#include "vocabulary.h"
#include "util.h"

namespace nplm {

shared_ptr<Corpus> readCorpus(
    const string& filename,
    const shared_ptr<Vocabulary>& vocab);

} // namespace nplm
