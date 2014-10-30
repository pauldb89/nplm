#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include "cdec_wrapper.h"

using namespace std;
using namespace nplm;
namespace po = boost::program_options;

void ParseOptions(
    const string& input, string& filename, string& feature_name,
    bool& normalized, bool& persistent_cache) {
  po::options_description options("NPLM language model options");
  options.add_options()
      ("file,f", po::value<string>()->required(),
          "File containing serialized language model")
      ("name,n", po::value<string>()->default_value("NPLM"),
          "Feature name")
      ("normalized", po::value<bool>()->required()->default_value(true),
          "Normalize the output of the neural network")
      ("persistent-cache",
          "Cache queries persistently between consecutive decoder runs");

  po::variables_map vm;
  vector<string> args;
  boost::split(args, input, boost::is_any_of(" "));
  po::store(po::command_line_parser(args).options(options).run(), vm);
  po::notify(vm);

  filename = vm["file"].as<string>();
  feature_name = vm["name"].as<string>();
  normalized = vm["normalized"].as<bool>();
  persistent_cache = vm.count("persistent-cache");
}

extern "C" FeatureFunction* create_ff(const string& str) {
  string filename, feature_name;
  bool normalized, persistent_cache;
  ParseOptions(
      str, filename, feature_name, normalized, persistent_cache);

  return new CdecNPLMWrapper(filename, feature_name, normalized, persistent_cache);
}
