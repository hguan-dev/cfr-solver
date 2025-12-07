#include "HandEvaluator.hpp"
#include <array>

static inline int convert_ID_to_eval_scheme(int old_id) {
  const int n = old_id - 1;
  const int rank = n % 13;
  const int suit = n / 13;
  return (rank * 4) + suit;
}

int HandEvaluator::hashQuinaryResult(
    const std::array<unsigned char, 13> &rankQuinary) {
  int sum = 0;
  int remainingCards = 7;

  for (std::size_t rank = 0; rank < rankQuinary.size(); rank++) {
    sum += DP_TABLE[rankQuinary[rank]][13 - rank - 1][remainingCards];
    remainingCards -= rankQuinary[rank];

    if (remainingCards <= 0)
      break;
  }

  return sum;
}

int HandEvaluator::evaluateHandRaw(const std::array<int, 2> &handIdx,
                                   const std::array<int, 5> &boardIdx) {

  const int a = convert_ID_to_eval_scheme(handIdx[0]);
  const int b = convert_ID_to_eval_scheme(handIdx[1]);
  const int c = convert_ID_to_eval_scheme(boardIdx[0]);
  const int d = convert_ID_to_eval_scheme(boardIdx[1]);
  const int e = convert_ID_to_eval_scheme(boardIdx[2]);
  const int f = convert_ID_to_eval_scheme(boardIdx[3]);
  const int g = convert_ID_to_eval_scheme(boardIdx[4]);

  int suit_hash = 0;

  suit_hash += bit_of_mod_4_x_3[a];
  suit_hash += bit_of_mod_4_x_3[b];
  suit_hash += bit_of_mod_4_x_3[c];
  suit_hash += bit_of_mod_4_x_3[d];
  suit_hash += bit_of_mod_4_x_3[e];
  suit_hash += bit_of_mod_4_x_3[f];
  suit_hash += bit_of_mod_4_x_3[g];

  if (SUITS_TABLE[suit_hash]) {
    int suit_binary[4] = {0};

    suit_binary[a & 0x3] |= bit_of_div_4[a];
    suit_binary[b & 0x3] |= bit_of_div_4[b];
    suit_binary[c & 0x3] |= bit_of_div_4[c];
    suit_binary[d & 0x3] |= bit_of_div_4[d];
    suit_binary[e & 0x3] |= bit_of_div_4[e];
    suit_binary[f & 0x3] |= bit_of_div_4[f];
    suit_binary[g & 0x3] |= bit_of_div_4[g];

    return FLUSH_TABLE[suit_binary[SUITS_TABLE[suit_hash] - 1]];
  }

  std::array<unsigned char, 13> quinary = {0};

  quinary[(a >> 2)]++;
  quinary[(b >> 2)]++;
  quinary[(c >> 2)]++;
  quinary[(d >> 2)]++;
  quinary[(e >> 2)]++;
  quinary[(f >> 2)]++;
  quinary[(g >> 2)]++;

  const int hash = hashQuinaryResult(quinary);

  return NOFLUSH_TABLE[hash];
}
