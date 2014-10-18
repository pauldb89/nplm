#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <cmath>
#include <string>
#include "../3rdparty/Eigen/Dense"

#include "util.h"

namespace nplm
{

// is this cheating?
using Eigen::Matrix;
using Eigen::MatrixBase;

enum activation_function_type {
    Tanh,
    HardTanh,
    Rectifier,
    Identity,
    Sigmoid,
    InvalidFunction
};

inline activation_function_type string_to_activation_function(
    const std::string &s) {
  if (s == "identity") {
    return Identity;
  } else if (s == "rectifier") {
    return Rectifier;
  } else if (s == "tanh") {
    return Tanh;
  } else if (s == "hardtanh") {
    return HardTanh;
  } else if (s == "sigmoid") {
    return Sigmoid;
  } else {
    return InvalidFunction;
  }
}

inline std::string activation_function_to_string(
    activation_function_type f) {
  if (f == Identity) {
    return "identity";
  } else if (f == Rectifier) {
    return "rectifier";
  } else if (f == Tanh) {
    return "tanh";
  } else if (f == HardTanh) {
    return "hardtanh";
  } else if (f == Sigmoid) {
    return "sigmoid";
  } else {
    cerr << "Invalid function type" << endl;
    exit(1);
  }
}

struct hardtanh_functor {
  double operator()(double x) const {
    if (x < -1.) {
      return -1.;
    } else if (x > 1.) {
      return 1.;
    } else {
      return x;
    }
  }
};

struct dhardtanh_functor {
  double operator()(double x) const {
    return x > -1. && x < 1. ? 1. : 0.;
  }
};

struct tanh_functor {
  double operator()(double x) const {
    return std::tanh(x);
  }
};

struct dtanh_functor {
  double operator()(double x) const {
    return 1-x*x;
  }
};

struct rectifier_functor {
  double operator()(double x) const {
    return std::max(x, 0.);
  }
};

struct drectifier_functor {
  double operator()(double x) const {
    return x > 0. ? 1. : 0.;
  }
};

struct sigmoid_functor {
  double operator()(double x) const {
    return 1 / (1 + exp(-x));
  }
};

struct dsigmoid_functor {
  double operator()(double x) const {
    double s = sigmoid(x);
    return s * (1 - s);
  }

  sigmoid_functor sigmoid;
};

class Activation_function {
 private:
  int size;
  activation_function_type f;

 public:
  Activation_function() : size(0), f(Rectifier) {}

  void resize(int size) {
    this->size = size;
  }

  void set_activation_function(activation_function_type f) {
    this->f = f;
  }

  template <typename Engine>
  void initialize(Engine &engine, bool init_normal, double init_range) {}

  int n_inputs() const {
    return size;
  }

  int n_outputs() const {
    return size;
  }

  template <typename DerivedIn, typename DerivedOut>
  void fProp(
      const MatrixBase<DerivedIn> &input,
      const MatrixBase<DerivedOut> &output) const {
    UNCONST(DerivedOut, output, my_output);

    switch (f) {
      case Identity:
        my_output = input;
        break;
      case Rectifier:
        my_output = input.unaryExpr(rectifier_functor());
        break;
      case Tanh:
        my_output = input.unaryExpr(tanh_functor());
        break;
      case HardTanh:
        my_output = input.unaryExpr(hardtanh_functor());
        break;
      case Sigmoid:
        my_output = input.unaryExpr(sigmoid_functor());
        break;
      default:
        cerr << "Unknown activation function" << endl;
        exit(1);
    }
  }

  template <typename DerivedGOut, typename DerivedGIn, typename DerivedIn, typename DerivedOut>
  void bProp(
      const MatrixBase<DerivedGOut> &input, MatrixBase<DerivedGIn> &output,
      const MatrixBase<DerivedIn> &finput,
      const MatrixBase<DerivedOut> &foutput) const {
    UNCONST(DerivedGIn, output, my_output);

    switch (f) {
      case Identity:
        my_output = input;
        break;
      case Rectifier:
        my_output = finput.array().unaryExpr(drectifier_functor()) * input.array();
        break;
      case Tanh:
        // Likely incorrect.
        my_output = foutput.array().unaryExpr(tanh_functor()) * input.array();
        break;
      case HardTanh:
        // Likely incorrect.
        my_output = finput.array().unaryExpr(hardtanh_functor()) * input.array();
        break;
      case Sigmoid:
        my_output = foutput.array().unaryExpr(dsigmoid_functor()) * input.array();
        break;
      default:
        cerr << "Unknown activation function" << endl;
        exit(1);
    }
  }
};

} // namespace nplm

#endif
