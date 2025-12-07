#include "Solver.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>

Solver::Solver(HandEvaluator &eval) : evaluator_(eval) {}

CFRNode *Solver::getNode(const std::string &infoSet, const GameState &state) {
  if (nodeMap_.find(infoSet) == nodeMap_.end()) {
    auto legal = state.getLegalActions();
    CFRNode node(legal.size());
    node.legalActions = legal;
    nodeMap_[infoSet] = node;
  }
  return &nodeMap_[infoSet];
}

double Solver::cfr(GameState &state, std::vector<int> &p0_cards,
                   std::vector<int> &p1_cards) {
  if (state.isTerminal()) {
    return state.getPayoff(p0_cards, p1_cards, evaluator_);
  }

  // Chance Sampling: Deal board cards if needed
  int cards_needed = 0;
  if (state.street == Street::FLOP && state.board.empty())
    cards_needed = 3;
  else if ((state.street == Street::TURN || state.street == Street::RIVER) &&
           state.board.size() < (size_t)state.street + 2)
    cards_needed = 1;

  if (cards_needed > 0) {
    // Draw distinct cards for the board
    for (int i = 0; i < cards_needed; ++i) {
      while (true) {
        int c = cardToInt(deck_.popTop());
        bool used = false;
        for (int x : p0_cards)
          if (x == c)
            used = true;
        for (int x : p1_cards)
          if (x == c)
            used = true;
        for (int x : state.board)
          if (x == c)
            used = true;

        if (!used) {
          state.board.push_back(c);
          break;
        }
      }
    }
  }

  int player = state.active_player;
  std::vector<int> &my_cards = (player == 0) ? p0_cards : p1_cards;
  std::string infoSet = state.getInfoSetKey(my_cards);
  CFRNode *node = getNode(infoSet, state);

  // Regret Matching to determine current strategy
  std::vector<double> strategy(node->legalActions.size());
  double regretSumTotal = 0.0;
  for (double r : node->regretSum)
    regretSumTotal += (r > 0) ? r : 0;

  for (size_t a = 0; a < node->legalActions.size(); ++a) {
    if (regretSumTotal > 0)
      strategy[a] =
          (node->regretSum[a] > 0) ? (node->regretSum[a] / regretSumTotal) : 0;
    else
      strategy[a] = 1.0 / node->legalActions.size();
  }

  // Recursive traversal
  std::vector<double> actionUtils(node->legalActions.size());
  double nodeUtil = 0.0;

  for (size_t a = 0; a < node->legalActions.size(); ++a) {
    GameState nextState = state;
    nextState.applyAction(node->legalActions[a]);

    actionUtils[a] = -cfr(nextState, p0_cards, p1_cards);
    nodeUtil += strategy[a] * actionUtils[a];
  }

  // Update Regrets and Average Strategy
  for (size_t a = 0; a < node->legalActions.size(); ++a) {
    double regret = actionUtils[a] - nodeUtil;
    node->regretSum[a] += regret;
    node->strategySum[a] += strategy[a];
  }

  return nodeUtil;
}

void Solver::train(int iterations) {
  for (int i = 0; i < iterations; ++i) {
    deck_.shuffle();
    std::vector<int> p0 = {cardToInt(deck_.popTop()),
                           cardToInt(deck_.popTop())};
    std::vector<int> p1 = {cardToInt(deck_.popTop()),
                           cardToInt(deck_.popTop())};

    GameState root;
    cfr(root, p0, p1);

    if ((i + 1) % 1000 == 0)
      std::cout << "Iteration " << (i + 1) << " complete..." << std::endl;
  }
}

void Solver::saveStrategy(const std::string &filename) {
  std::ofstream file(filename);
  file << "InfoSet,Fold,CheckCall,BetRaise\n";

  for (auto &pair : nodeMap_) {
    CFRNode &node = pair.second;
    double sum = 0;
    for (double s : node.strategySum)
      sum += s;

    std::vector<double> avg(node.legalActions.size());
    for (size_t i = 0; i < node.legalActions.size(); ++i) {
      avg[i] = (sum > 0) ? (node.strategySum[i] / sum)
                         : (1.0 / node.legalActions.size());
    }

    // Map to standard columns
    double pFold = 0, pCall = 0, pRaise = 0;
    for (size_t i = 0; i < node.legalActions.size(); ++i) {
      if (node.legalActions[i].type == ActionType::FOLD)
        pFold = avg[i];
      else if (node.legalActions[i].type == ActionType::CHECK_CALL)
        pCall = avg[i];
      else if (node.legalActions[i].type == ActionType::BET_RAISE)
        pRaise = avg[i];
    }

    file << pair.first << "," << pFold << "," << pCall << "," << pRaise << "\n";
  }
  file.close();
  std::cout << "Strategy saved to " << filename << std::endl;
}
