#pragma once

#include "Deck.hpp"
#include "GameState.hpp"
#include <vector>

struct CFRNode {
  std::vector<double> regretSum;
  std::vector<double> strategySum;
  std::vector<Action> legalActions;

  CFRNode() {}
  CFRNode(int num_actions)
      : regretSum(num_actions, 0.0), strategySum(num_actions, 0.0) {}
};

class Solver {
public:
  Solver(HandEvaluator &evaluator);

  // Run N iterations of External Sampling MCCFR
  void train(int iterations);

  // Export the average strategy to a CSV file
  void saveStrategy(const std::string &filename);

private:
  HandEvaluator &evaluator_;
  std::unordered_map<std::string, CFRNode> nodeMap_;
  Deck deck_;

  // Recursive MCCFR function
  double cfr(GameState &state, std::vector<int> &p0_cards,
             std::vector<int> &p1_cards);

  // Helper to get or create a node in the map
  CFRNode *getNode(const std::string &infoSet, const GameState &state);
};
