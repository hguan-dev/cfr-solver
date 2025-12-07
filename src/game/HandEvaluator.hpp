#pragma once

#include "HandHashes.hpp"
#include <array>

class HandEvaluator {
public:
  int evaluateHandRaw(const std::array<int, 2> &hole,
                      const std::array<int, 5> &board);

  int hashQuinaryResult(const std::array<unsigned char, 13> &rankQuinary);
};
