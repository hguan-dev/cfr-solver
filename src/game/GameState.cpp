#include "GameState.hpp"
#include <algorithm>

int cardToInt(const Card &c) {
  return static_cast<int>(c.getSuit()) * 13 + static_cast<int>(c.getRank());
}

GameState::GameState() {
  street = Street::PREFLOP;
  pot = SMALL_BLIND + BIG_BLIND;
  stack = {STACK_SIZE - SMALL_BLIND, STACK_SIZE - BIG_BLIND};
  bets = {SMALL_BLIND, BIG_BLIND};
  active_player = 0;
  board.reserve(5);
  history = "";
  raises_this_street = 0;
  is_folded = false;
}

uint64_t
GameState::getInfoSetKeyHash(const std::vector<int> &hole_cards) const {
  std::vector<int> sorted = hole_cards;
  if (sorted[0] > sorted[1])
    std::swap(sorted[0], sorted[1]);

  const uint64_t P1 = 31;
  const uint64_t P2 = 59;

  uint64_t hash = 0;

  hash = (hash * P1) + sorted[0]; // Combine Card 1
  hash = (hash * P1) + sorted[1]; // Combine Card 2

  for (int c : board) {
    hash = (hash * P1) + c;
  }

  hash = (hash * P2) + history.length(); // Add length first
  for (char h : history) {
    hash = (hash * P2) + (uint64_t)h;
  }

  return hash;
}

bool GameState::isTerminal() const {
  return is_folded || street == Street::SHOWDOWN;
}

FixedActions GameState::getLegalActions() const {
  FixedActions actions; // Allocated on stack (Instant)

  if (isTerminal()) {
    return actions; // Returns count = 0
  }

  double to_call = bets[1 - active_player] - bets[active_player];

  // 1. Fold Option
  // Only available if we are facing a bet (to_call > 0)
  if (to_call > 0) {
    actions.push_back({ActionType::FOLD, 0.0});
  }

  // 2. Check/Call Option
  // Always valid (Check if to_call == 0, Call if to_call > 0)
  actions.push_back({ActionType::CHECK_CALL, to_call});

  // 3. Bet/Raise Option
  // Valid if we haven't hit the raise limit AND we have enough chips
  if (raises_this_street < MAX_RAISES) {
    double bet_amount =
        (street == Street::PREFLOP || street == Street::FLOP) ? 1.0 : 2.0;

    // Ensure we have enough stack to cover the call + the raise
    if (stack[active_player] >= to_call + bet_amount) {
      actions.push_back({ActionType::BET_RAISE, to_call + bet_amount});
    }
  }

  return actions;
}

void GameState::applyAction(const Action &action) {
  if (action.type == ActionType::FOLD) {
    is_folded = true;
    history += "f";
    return;
  }

  stack[active_player] -= action.amount;
  bets[active_player] += action.amount;
  pot += action.amount;

  if (action.type == ActionType::BET_RAISE) {
    history += "r";
    raises_this_street++;
    active_player = 1 - active_player;
  } else {
    history += (action.amount > 0) ? "c" : "k";

    if (bets[0] == bets[1]) {
      bool street_over = false;

      if (street == Street::PREFLOP) {
        // Preflop: Round ends if bets > 1BB (raise called) or BB checks option
        if (bets[0] == BIG_BLIND)
          street_over = (active_player == 1);
        else
          street_over = true;
      } else {
        // Postflop: Round ends if P0 (Dealer) acts, or P1 calls a bet
        if (active_player == 0)
          street_over = true;
        else
          street_over = (action.amount > 0);
      }

      if (street_over)
        nextStreet();
      else
        active_player = 1 - active_player;
    } else {
      active_player = 1 - active_player;
    }
  }
}

void GameState::nextStreet() {
  active_player = 1;
  bets = {0.0, 0.0};
  raises_this_street = 0;

  if (street == Street::PREFLOP)
    street = Street::FLOP;
  else if (street == Street::FLOP)
    street = Street::TURN;
  else if (street == Street::TURN)
    street = Street::RIVER;
  else if (street == Street::RIVER)
    street = Street::SHOWDOWN;

  if (street != Street::SHOWDOWN)
    history += "/";
}

std::string GameState::getInfoSetKey(const std::vector<int> &hole_cards) const {
  // 1. Sort Hole Cards (Canonicalization)
  std::vector<int> sorted = hole_cards;
  if (sorted[0] > sorted[1])
    std::swap(sorted[0], sorted[1]);

  std::string key;
  key.reserve(128);

  // 3. Build Key: Hole Cards | Board | History

  // Append Hole Cards: Card1_Card2
  key += std::to_string(sorted[0]);
  key += "_";
  key += std::to_string(sorted[1]);
  key += "|";

  for (int c : board) {
    key += std::to_string(c);
    key += "_";
  }
  key += "|";

  key += history;

  return key;
}

double GameState::getPayoff(const std::vector<int> &p0,
                            const std::vector<int> &p1,
                            HandEvaluator &eval) const {
  double p0_in = 100.0 - stack[0];

  if (is_folded) {
    return (active_player == 0) ? -p0_in : (100.0 - stack[1]);
  }

  if (street == Street::SHOWDOWN) {
    std::array<int, 2> h0 = {p0[0], p0[1]};
    std::array<int, 2> h1 = {p1[0], p1[1]};

    std::array<int, 5> b;

    for (size_t i = 0; i < 5; ++i)
      b[i] = board[i];

    int s0 = eval.evaluateHandRaw(h0, b);
    int s1 = eval.evaluateHandRaw(h1, b);

    if (s0 > s1)
      return pot - p0_in;
    if (s0 < s1)
      return -p0_in;
  }
  return 0.0;
}
