# Heads-Up Limit Texas Hold'em CFR Solver

**Authors:** Xiyang Liu, Harry Guan

## Overview
This project implements a high-performance Heads-Up Limit Hold'em solver using Monte Carlo Counterfactual Regret Minimization (MCCFR), coupled with a Machine Learning pipeline to predict optimal strategy distributions. 

Texas Hold'em is computationally intractable for brute-force methods due to a massive game tree exceeding 10^70 states. Counterfactual Regret Minimization (CFR) solves this by iterative self-play, minimizing regret to converge on an unexploitable Nash Equilibrium. However, querying these massive solution sets is slow. 

To solve this, we built a custom C++ MCCFR engine to generate ground-truth data, which is then fed into a Python ML pipeline. Instead of simple classification, the model predicts the optimal action probability distribution (mixed strategies), capturing the necessary unpredictability of optimal play. This compresses the massive solution set into a lightweight function, enabling instant real-time inference.

## Architecture

### 1. C++ Monte Carlo CFR Engine
The data generation step is handled by a custom C++ MCCFR engine built from scratch to maximize computational throughput. 
* **Zero-Allocation Optimization:** We engineered "zero-allocation" hot paths by replacing dynamic vectors and strings with fixed-size arrays and pre-allocated memory pools.
* **High-Throughput Simulation:** The solver achieves >50,000 iterations/second via integer-based state hashing and fast hand evaluation.
* **State Abstraction:** Currently utilizes a temporary 100-fixed-board abstraction to maintain tractability.
* **Data Export:** Serializes converged strategy distributions directly into feature vectors (`strategy_output.csv`) to create a robust dataset for regression.

### 2. Python Machine Learning Pipeline
The ML pipeline predicts the CFR distributions (Fold, Check/Call, Bet/Raise probabilities) using features extracted from the game state. 
* **Feature Engineering (Polars):** The initial 8 raw features (hole cards, community cards, history) are expanded into 58 highly predictive features. These capture hand strength through card interactions, rank gaps, board texture (suited/connected), and opponent aggression ratios.
* **Model Training:** Evaluates multiple architectures against baselines (Uniform and Marginal probabilities):
  * **Constrained Linear Model:** 3 separate Ridge regression models with normalization.
  * **Boosted Tree (XGBoost):** Trains a classifier and uses the softmax probabilities as a proxy for the true distribution.
  * **Neural Network (PyTorch):** Predicts the three probabilities using regression with ReLU activation, applying a softmax function at the output layer.
* **Cost Function:** The models are trained and evaluated primarily using KL Divergence.

## Results

Trained on a dataset of 2,216,961 samples with an 80-20 train-test split, the nonlinear models significantly outperformed the linear models.

| Metric | Uniform Baseline | Marginal Heuristic | 3x Lasso | Boosted Tree | Neural Network |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Average KL Div** | 0.932 | 0.735 | 0.530 | 0.285 | 0.274 |
| **Average TVD** | 0.613 | 0.491 | 0.386 | 0.206 | 0.226 |
| **Fold MAE** | 0.404 | 0.392 | 0.299 | 0.175 | 0.186 |
| **Check/Call MAE** | 0.490 | 0.433 | 0.370 | 0.197 | 0.213 |
| **Bet/Raise MAE** | 0.332 | 0.156 | 0.103 | 0.039 | 0.052 |

The **Boosted tree** demonstrated the best performance on Total Variation Distance (TVD) and Mean Absolute Error (MAE). The **Neural network** achieved the lowest KL divergence.

## Getting Started

### Prerequisites
* CMake (C++ Build System)
* C++17 or higher
* Python 3.10+
* `pip install polars scikit-learn xgboost torch joblib numpy`

### Running the C++ Solver
```bash
mkdir build && cd build
cmake ..
make
./cfr_solver
```
This will run the MCCFR iterations and generate strategy_output.csv in your working directory.

Running the ML Pipeline
Ensure your dataset (strategy_output_small.csv or the full output) is in the root directory, then run:

```bash
python main.py
```
This will extract features, split the data, train all models sequentially, and output performance metrics to the console. Saved models will be exported as .joblib files.

### Future Work
* **Ensembling**: Combine the Boosted Tree and Neural Network, as both models have their own strengths compared to the other.
* **Compute Scaling:** Implement multithreading in the C++ engine to handle larger state spaces and remove the 100-board abstraction.
* **Feature Refinement:** Investigate feature importance, as current features likely do not contribute equal predictive power. Experiment with kernel functions to better capture the nonlinear relationship between features and the CFR distribution.
