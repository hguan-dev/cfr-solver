#pragma once

#include "../cards/Deck.hpp"
#include "../game/GameState.hpp"
#include <string>
#include <unordered_map>
#include <vector>

struct TrainingScenario {
  std::vector<int> p0_cards;
  std::vector<int> p1_cards;
  std::vector<int> board;
};

struct CFRNode {
  std::vector<double> regretSum;
  std::string infoSetKeyString;
  std::vector<double> strategySum;
  FixedActions legalActions;
  int visits; // Track visits for pruning/debugging

  CFRNode() : visits(0) {}
  CFRNode(int num_actions)
      : regretSum(num_actions, 0.0), strategySum(num_actions, 0.0), visits(0) {}
};

class Solver {
public:
  Solver(HandEvaluator &evaluator);

  void train(int iterations);
  void saveStrategy(const std::string &filename);

private:
  HandEvaluator &evaluator_;
  std::unordered_map<uint64_t, CFRNode> nodeMap_;
  Deck deck_;

  std::vector<TrainingScenario> scenarios_;
  std::vector<int> iteration_board_;

  double cfr(GameState &state, std::vector<int> &p0_cards,
             std::vector<int> &p1_cards);
  CFRNode *getNode(uint64_t infoSetHash, const GameState &state,
                   const std::vector<int> &myCards);
};
