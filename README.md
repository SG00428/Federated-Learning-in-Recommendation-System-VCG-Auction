# Federated Learning in Recommendation System with VCG Auction

This project implements a **Federated Learning (FL)** framework combined with an **exact VCG (Vickrey–Clarke–Groves) auction** mechanism for client selection under a **global budget constraint**.

Client selection is formulated as a **Winner Determination Problem (WDP)** (knapsack), solved exactly using **Integer Linear Programming (ILP)**. VCG payments are computed via the **Clarke pivot rule** by re-solving the ILP with each winning client removed.

The project supports:
- **Synthetic data simulation**
- **MovieLens 100k / 1M datasets**
- **Exact VCG payments**
- **Neural Collaborative Filtering (NCF)** using PyTorch
- **FedAvg aggregation**
- **OR-Tools or PuLP** as ILP solvers

---

##  Key Features

- **Federated Learning**
  - Client-side local training
  - Server-side FedAvg aggregation
- **Auction-based Client Selection**
  - Exact knapsack solution via ILP
  - Budget-constrained selection
- **VCG Payments**
  - Truthful, incentive-compatible payments
  - Exact Clarke pivot computation
- **Recommender System**
  - Neural Collaborative Filtering (GMF + MLP)
- **Evaluation**
  - Test RMSE over rounds
  - Cumulative payments and utilities
- **Datasets**
  - Synthetic data generator
  - MovieLens 100k / 1M loader

---

## Experiment Configuration

Certain parameters can be modified near the bottom of the file:

use_movielens = False        # True → MovieLens, False → Synthetic

ml_dataset = '100k'         # '100k' or '1M'

sample_users = None         # Limit number of users (for speed)

candidate_pool_size = 30

rounds = 8

budget = 40.0

time_limit_ilp = 30.0


## Output & Results

After execution, the following are generated:

Selected clients per round

VCG payments and utilities

RMSE before and after each round

Cumulative payments plot


## Auction & Economic Details

Bid format per client:
(declared_value, declared_cost)

Winner determination:
Exact ILP knapsack under global budget

Payment rule:
Clarke pivot (VCG)

Utility (profit):

utility_i = payment_i - cost_i


Only individually rational (payment ≥ cost) clients are kept.

##  Evaluation Metric

RMSE (Root Mean Squared Error) on held-out test data

Computed before and after each FL round

