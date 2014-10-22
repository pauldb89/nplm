#pragma once

#include <memory>
#include <string>

#include "config.h"
#include "vocabulary.h"
#include "util.h"

namespace nplm {

MatrixInt ExtractMinibatch(
    const shared_ptr<Corpus>& corpus,
    const shared_ptr<Vocabulary>& vocab,
    const Config& config,
    data_size_t start_index);


shared_ptr<Corpus> readCorpus(
    const string& filename,
    const shared_ptr<Vocabulary>& vocab);

} // namespace nplm
