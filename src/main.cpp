#include "cfr/Solver.hpp"
#include "game/HandEvaluator.hpp"
#include <chrono>
#include <iostream>

int main() {
  std::cout << "--- Heads-Up Limit Hold'em Solver ---" << std::endl;
  std::cout << "[1] Initializing Hand Evaluator..." << std::endl;
  HandEvaluator evaluator;

  std::cout << "[2] Initializing CFR Solver..." << std::endl;
  Solver solver(evaluator);

  int iterations = 100000;

  std::cout << "[3] Starting MCCFR for " << iterations << " iterations..."
            << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  solver.train(iterations);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  std::cout << "   -> Finished in " << elapsed.count() << "s" << std::endl;
  std::cout << "   -> Speed: " << (iterations / elapsed.count()) << " iter/s"
            << std::endl;

  std::string filename = "strategy_output.csv";
  std::cout << "[4] Exporting strategy to " << filename << "..." << std::endl;
  solver.saveStrategy(filename);

  std::cout << "Done." << std::endl;
  return 0;
}
