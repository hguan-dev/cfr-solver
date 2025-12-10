#include "Solver.hpp"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cstring>

Solver::Solver(HandEvaluator &eval) : evaluator_(eval) {}

CFRNode *Solver::getNode(uint64_t hash, const GameState &state,
                         const std::vector<int> &my_cards) {
  auto it = nodeMap_.find(hash);

  if (it == nodeMap_.end()) {
    auto &node = nodeMap_[hash];

    std::vector<int> sorted = my_cards;
    if (sorted[0] > sorted[1])
      std::swap(sorted[0], sorted[1]);
    node.key.c1 = static_cast<uint8_t>(sorted[0]);
    node.key.c2 = static_cast<uint8_t>(sorted[1]);
    std::memset(node.key.board, 0, 5);
    for (size_t i = 0; i < state.board.size(); ++i) {
      node.key.board[i] = static_cast<uint8_t>(state.board[i]);
    }

    int len = std::min((int)sizeof(node.key.history) - 1, state.history_length);
    std::memcpy(node.key.history, state.history, len);
    node.key.history[len] = '\0';

    node.legalActions = state.getLegalActions();
    node.regretSum.resize(node.legalActions.size(), 0.0);
    node.strategySum.resize(node.legalActions.size(), 0.0);
    return &node;
  }
  return &it->second;
}

double Solver::cfr(GameState &state, std::vector<int> &p0_cards,
                   std::vector<int> &p1_cards) {
  if (state.isTerminal()) {
    return state.getPayoff(p0_cards, p1_cards, evaluator_);
  }

  if (state.street == Street::FLOP && state.board.empty()) {
    state.board.push_back(iteration_board_[0]);
    state.board.push_back(iteration_board_[1]);
    state.board.push_back(iteration_board_[2]);
  } else if (state.street == Street::TURN && state.board.size() == 3) {
    state.board.push_back(iteration_board_[3]);
  } else if (state.street == Street::RIVER && state.board.size() == 4) {
    state.board.push_back(iteration_board_[4]);
  }

  int player = state.active_player;
  std::vector<int> &my_cards = (player == 0) ? p0_cards : p1_cards;
  uint64_t infoSetHash = state.getInfoSetKeyHash(my_cards);
  CFRNode *node = getNode(infoSetHash, state, my_cards);

  node->visits++;

  // Regret Matching
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

  std::vector<double> actionUtils(node->legalActions.size());
  double nodeUtil = 0.0;

  for (size_t a = 0; a < node->legalActions.size(); ++a) {
    GameState nextState = state;
    nextState.applyAction(node->legalActions[a]);

    actionUtils[a] = -cfr(nextState, p0_cards, p1_cards);
    nodeUtil += strategy[a] * actionUtils[a];
  }

  for (size_t a = 0; a < node->legalActions.size(); ++a) {
    double regret = actionUtils[a] - nodeUtil;
    node->regretSum[a] += regret;
    node->strategySum[a] += strategy[a];
  }

  return nodeUtil;
}

void Solver::train(int iterations) {
  if (scenarios_.empty()) {
    nodeMap_.reserve(500000000);

    std::cout << "[Solver] Generating 100 fixed BOARDS..." << std::endl;
    for (int i = 0; i < 100; ++i) {
      deck_.shuffle();
      TrainingScenario s;
      deck_.popTop();
      deck_.popTop();
      deck_.popTop();
      deck_.popTop();

      for (int k = 0; k < 5; ++k)
        s.board.push_back(deck_.popTop());
      scenarios_.push_back(s);
    }
  }

  for (int i = 0; i < iterations; ++i) {
    const auto &s = scenarios_[i % scenarios_.size()];
    iteration_board_ = s.board;

    deck_.shuffle();
    std::vector<int> p0, p1;

    while (p0.size() < 2) {
      int c = deck_.popTop();
      bool collision = false;
      for (int b : iteration_board_)
        if (b == c)
          collision = true;
      if (!collision)
        p0.push_back(c);
    }

    while (p1.size() < 2) {
      int c = deck_.popTop();
      bool collision = false;
      for (int b : iteration_board_)
        if (b == c)
          collision = true;
      for (int p : p0)
        if (p == c)
          collision = true;
      if (!collision)
        p1.push_back(c);
    }

    GameState root;
    cfr(root, p0, p1);

    if ((i + 1) % 50000 == 0)
      std::cout << "Iteration " << (i + 1) << " complete..." << std::endl;
  }
}

void Solver::saveStrategy(const std::string &filename) {
  std::ofstream file(filename);

  file << "C1,C2,B1,B2,B3,B4,B5,History,Fold,CheckCall,BetRaise\n";

  int saved_count = 0;
  for (auto &pair : nodeMap_) {
    CFRNode &node = pair.second;

    // Node pruning before data
    if (node.visits < 25)
      continue;
    saved_count++;

    double sum = 0;
    for (double s : node.strategySum)
      sum += s;

    std::vector<double> avg(node.legalActions.size());
    for (size_t i = 0; i < node.legalActions.size(); ++i) {
      avg[i] = (sum > 0) ? (node.strategySum[i] / sum)
                         : (1.0 / node.legalActions.size());
    }

    double pFold = 0, pCall = 0, pRaise = 0;
    for (size_t i = 0; i < node.legalActions.size(); ++i) {
      if (node.legalActions[i].type == ActionType::FOLD)
        pFold = avg[i];
      else if (node.legalActions[i].type == ActionType::CHECK_CALL)
        pCall = avg[i];
      else if (node.legalActions[i].type == ActionType::BET_RAISE)
        pRaise = avg[i];
    }

    file << (int)node.key.c1 << "," << (int)node.key.c2 << ",";
    file << (int)node.key.board[0] << "," << (int)node.key.board[1] << ","
         << (int)node.key.board[2] << "," << (int)node.key.board[3] << ","
         << (int)node.key.board[4] << ",";
    file << node.key.history << ",";

    file << pFold << "," << pCall << "," << pRaise << "\n";
  }
  file.close();
  std::cout << "Strategy saved to " << filename << " (" << saved_count
            << " rows)" << std::endl;
}
