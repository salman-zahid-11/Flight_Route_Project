# Flight Route Planning on the OpenFlights Route Network

**A\* Search · Genetic Algorithm · Forward Checking**

Artificial Intelligence course project, Dept. of ICT.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salman-zahid-11/Flight_Route_Project/blob/main/AI_Project_Flight_Routes.ipynb)


| | |
|---|---|
| **Students** | Md. Salman Zahid (23549009011), Md. Nasimuzzaman Emon (2254901045), Ferdous Hasan Nahid (23549009061) |
| **Instructor** | Jasia Fairiz Raisa, Lecturer, Dept. of ICT |
| **Dataset** | OpenFlights Flight Route Database (Kaggle mirror) |
| **Language** | Python 3 (Jupyter / Google Colab) |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dataset](#2-dataset)
3. [What is extracted from the dataset](#3-what-is-extracted-from-the-dataset)
4. [Algorithms and methodology](#4-algorithms-and-methodology)
5. [How to run](#5-how-to-run)
6. [Results and performance analysis](#6-results-and-performance-analysis)
7. [Comparison of the algorithms](#7-comparison-of-the-algorithms)
8. [Which algorithm is best? Final recommendation](#8-which-algorithm-is-best-final-recommendation)
9. [Tools and technologies](#9-tools-and-technologies)
10. [Repository structure](#10-repository-structure)
11. [Limitations and future work](#11-limitations-and-future-work)
12. [References](#12-references)

---

## 1. Overview

Airline routes form a natural graph: **airports are nodes** and **scheduled flights are directed edges**. This project models the OpenFlights route database as such a graph and solves three practical questions, each with one AI technique:

| # | Real-world question | AI formulation | Algorithm |
|---|---|---|---|
| 1 | What is the shortest way to fly from airport A to airport B? | Single-pair shortest path on a weighted directed graph | **A\* search** |
| 2 | In which order should K cities be visited so the round trip is shortest? | Asymmetric travelling-salesman problem | **Genetic Algorithm** |
| 3 | Can busy airports get one of k slots so that airports with a direct route never share a slot? | Constraint satisfaction / graph colouring | **Backtracking with Forward Checking** |

The project then **analyses the performance** of the three algorithms (accuracy, run time, memory, nodes expanded) and ends with a **recommendation** on which algorithm to use and when.

**Headline results**

- **A\*** found the optimal path in **100 %** of 300 random queries, expanding **88 % fewer airports** than Dijkstra and running about **2.6× faster**.
- **Genetic Algorithm** reached **99.77 %** of the exact optimum on a 16-city tour (optimum found in **18 of 20** runs) and beat nearest-neighbour + 2-opt by **5.1 %** on 30 cities.
- **Forward Checking + MRV** assigned slots to **100 airports** with **0 violations** in about 100 assignments (about 4 ms), while plain backtracking and plain forward checking exhausted the 200,000-assignment budget from 40 airports onward.
- **Best algorithm for the flight-route problem: A\*.**

---

## 2. Dataset

- **Source:** OpenFlights Flight Route Database (Kaggle mirror) plus the OpenFlights airport table for coordinates.
  - OpenFlights: <https://openflights.org/data.html> (Open Database Licence)
  - Kaggle: <https://www.kaggle.com/datasets/openflights/flight-route-database>
- **Raw size:** 67,663 route records.
- **Route columns:** `airline, airline_id, src, src_id, dst, dst_id, codeshare, stops, equipment`.
- **Airport columns used:** `id, iata, icao, country, lat, lon`.

### Cleaning pipeline (row-count log)

| Step | Rows remaining | Rows removed |
|---|---:|---:|
| Raw rows | 67,663 | 0 |
| Drop rows with missing source/destination airport id | 67,240 | 423 |
| Drop routes whose airports are unknown / have bad coordinates | 66,771 | 469 |
| Remove self-loops (src = dst) | 66,770 | 1 |
| Unique directed airport pairs (graph edges) | 36,906 | 29,864 |

Missing values written as `\N` are converted to `NaN`.

### Graph statistics after cleaning

| Statistic | Value |
|---|---:|
| Airports (nodes) | 3,214 |
| Routes (edges) | 36,906 |
| Airlines | 566 |
| Countries | 225 |
| Max degree | 248 |
| Mean route length | 1,760 km |
| Longest route | 16,082 km |
| Largest strongly connected component | 3,147 airports (97.9 %) |

The route graph is directed and not fully connected, so **all experiments use the largest strongly connected component**; this guarantees that every query has an answer.

---

## 3. What is extracted from the dataset

The raw files contain more fields than the three problems need. For each problem only the relevant parts were extracted:

| Source field(s) | Used for | Extracted data structure |
|---|---|---|
| routes: `src_id`, `dst_id` | Graph edges (all problems) | 36,906 unique directed airport pairs |
| airports: `id`, `lat`, `lon` | Edge weights and A\* heuristic | Haversine distance (km) per edge; coordinate arrays for h(n) |
| connectivity | Every query has an answer | Largest strongly connected component |
| node degree, country | GA city selection | K = 16 and K = 30 cities (degree ≥ 15, at most one city per country); asymmetric cost matrix from all-pairs shortest paths |
| node degree, undirected links | Forward Checking | Conflict graph of the n busiest airports (n = 10 … 100); k from the DSATUR colouring bound |
| routes: `airline`; airports: `country` | Descriptive statistics only | 566 airlines, 225 countries |
| `codeshare`, `stops`, `equipment`, altitude, timezone, DST | **Not used** | Excluded: no effect on a distance-based cost |

**Cost model:** the cost of a flight is the great-circle (haversine) distance in kilometres between its two airports.

---

## 4. Algorithms and methodology

### 4.1 A\* search: shortest route

- **State:** current airport. **Action:** take a direct route. **Goal:** the destination airport. **Cost:** sum of haversine distances.
- Nodes are ordered by `f(n) = g(n) + h(n)`, where `g` is the distance already flown and `h` is the straight-line (haversine) distance still to go.
- The open list is a binary heap; a closed set prevents re-expansion.
- **Heuristic properties (verified on the real graph):**
  - *Admissible*: `h` never overestimates, because a flight cannot be shorter than the straight line. Checked: 0 violations in 300 queries.
  - *Consistent*: edge weights are haversine distances, so the triangle inequality gives `h(n) ≤ w(n,m) + h(m)`. Checked: 0 violations in 184,530 edge checks.
  - Heuristic tightness `h / true cost`: mean 0.869, min 0.261.
- Because `h` is consistent, the first time the goal leaves the open list its path is optimal.
- **Baselines implemented for comparison:** Dijkstra (`h = 0`), Weighted A\* (`w = 1.5`) and BFS (fewest flights).

### 4.2 Genetic Algorithm: multi-city tour

- **Chromosome:** a permutation of the K−1 non-start cities. **Fitness:** length of the closed tour.
- The cost between two cities is the **shortest distance through the real route network**, so the cost matrix is asymmetric.
- **Operators:** tournament selection (size 3), Order Crossover (probability 0.9), swap/inversion mutation (probability 0.25), elitism (best 2 tours always survive).
- **Settings:** population 200, 300 generations (30-city run: population 300, 600 generations).
- **Validation:** exact optimum from **Held–Karp dynamic programming** on 16 cities; heuristic baselines on 30 cities (nearest neighbour, nearest neighbour + 2-opt, random search with the same number of evaluations).

### 4.3 Backtracking with Forward Checking: slot assignment

- **Variables:** airports. **Domains:** slots `{0, …, k−1}`. **Constraints:** every route is a "not-equal" constraint.
- After assigning a slot to an airport, that slot is removed from the domains of its unassigned neighbours. An empty domain (wipe-out) triggers immediate backtracking.
- Domains are stored as **bit-masks** for speed.
- **Three solvers compared:** plain backtracking (BT), Forward Checking (FC), and FC with the **Minimum-Remaining-Values (MRV)** heuristic with degree as tie-breaker.
- Each run stops after a budget of **200,000 assignments**. `k` equals the DSATUR colouring bound, so a solution is guaranteed to exist. Every solution is verified by an independent function that counts violated constraints.

---

## 5. How to run

### Option A: Google Colab (recommended)

1. Click the **Open in Colab** badge at the top (or upload `AI_Project_Flight_Routes.ipynb` to <https://colab.research.google.com>).
2. Choose **Runtime → Run all**.
3. The notebook downloads the OpenFlights `routes.dat` and `airports.dat` automatically. If you want to use your own `routes.csv`, upload it to the Colab file panel before running.

### Option B: Local Jupyter

```bash
pip install numpy pandas scipy networkx matplotlib jupyter
jupyter notebook AI_Project_Flight_Routes.ipynb
```

Put `routes.dat` (or `routes.csv`) and `airports.dat` in the same folder, or let the notebook download them.

### Run time and settings

| Mode | How | Time |
|---|---|---|
| Full experiments (as in the report) | `QUICK = False` (default, first code cell) | about 3–5 minutes |
| Quick test | `QUICK = True` | about 1 minute |

The random seed is fixed to **42**. Small differences from the numbers below can appear if the data version differs or on a different machine.

### What the notebook contains

1. Setup and data loading
2. Cleaning with a row-count log, graph construction
3. A\* search: benchmark on random queries, heuristic verification, example routes, search visualisation (JFK → SYD)
4. Genetic Algorithm: 16-city validation against Held–Karp, 30-city baselines, convergence plots, hyper-parameter grid
5. Forward Checking: BT vs FC vs FC + MRV, smallest feasible number of slots
6. Final performance summary and recommendation

---

## 6. Results and performance analysis

### 6.1 A\* (300 random queries)

| Metric | A\* | Weighted A\* (w=1.5) | Dijkstra | BFS |
|---|---:|---:|---:|---:|
| Optimal paths (%) | **100** | 36.30 | 100 | 14.70 |
| Path validity (%) | 100 | 100 | 100 | 100 |
| Mean cost (km) | 9,581.30 | 9,840 | 9,581.30 | 11,352.80 |
| Cost overhead vs optimum (%) | 0 | 2.70 | 0 | 18.49 |
| Mean hops | 4.91 | 4.73 | 4.91 | 3.91 |
| Mean nodes expanded | **181** | 54.80 | 1,518.90 | 1,538.10 |
| Mean time (ms) | **1.75** | 0.719 | 4.58 | 2.53 |
| Peak memory (KiB) | **90.5** | 43.5 | 322.8 | 181.9 |

- A\* and Dijkstra are both optimal, but A\* expands **88 % fewer nodes**, is **about 2.6× faster** and uses **about 72 % less peak memory**.
- BFS minimises the number of flights (3.9 vs 4.9) but its paths are on average **18.5 % longer**.
- Weighted A\* is the fastest (about 55 nodes) but pays 2.7 % extra distance and is optimal in only 36.3 % of queries: a speed versus optimality trade-off.
- Example: New York (JFK) → Sydney (SYD): Dijkstra expands **2,983** airports, A\* only **21**. Optimal route JFK → LAX → SYD, 16,035 km.

### 6.2 Genetic Algorithm

**16 cities (exact optimum 54,365.9 km from Held–Karp, 20 runs)**

| Metric | Value |
|---|---:|
| GA best / mean / worst (km) | 54,365.9 / 54,496.6 / 55,672.1 |
| Std (km) | 391.86 |
| Mean solution quality | **99.77 %** |
| Optimum found | **18 / 20 runs** |
| Mean generations to best | 91.6 |
| Mean time per run | 1.77 s |

**30 cities (no exact optimum available)**

| Method | Tour cost (km) | Quality vs best known |
|---|---:|---:|
| Genetic algorithm (best of 10 runs) | 88,256.50 | 100 % |
| Genetic algorithm (mean of 10 runs) | 90,880.40 | 97.11 % |
| Nearest neighbour | 98,798.40 | 89.33 % |
| Nearest neighbour + 2-opt | 93,015.70 | 94.88 % |
| Random search (same evaluations) | 185,839.30 | 47.49 % |

The best GA tour is **5.1 % shorter** than nearest neighbour + 2-opt.

**Hyper-parameters:** mutation rates of 0.25–0.5 reached the optimum in almost every run; 0.05 sometimes stalled in local optima for small populations. The one dip (population 200, mutation 0.25: 4/5) is within the noise of only five seeds per setting.

### 6.3 Backtracking vs Forward Checking vs FC + MRV

n busiest airports, budget 200,000 assignments (selected rows):

| n | k | Solver | Status | Assignments | Time (s) | Violations |
|---:|---:|---|---|---:|---:|---:|
| 30 | 15 | BT | solved | 1,150 | 0.001 | 0 |
| 30 | 15 | FC | solved | 44 | 0.0002 | 0 |
| 30 | 15 | FC + MRV | solved | 30 | 0.0003 | 0 |
| 40 | 17 | BT | limit | 200,001 | 0.17 | n/a |
| 40 | 17 | FC | limit | 200,001 | 0.50 | n/a |
| 40 | 17 | FC + MRV | solved | 40 | 0.001 | 0 |
| 100 | 22 | BT | limit | 200,001 | 0.24 | n/a |
| 100 | 22 | FC | limit | 200,001 | 0.78 | n/a |
| 100 | 22 | FC + MRV | **solved** | **100** | **0.0037** | **0** |

- **FC + MRV solved every size with about n assignments and no violations.**
- Plain backtracking failed from n = 40; plain forward checking failed at n = 40, 50, 80 and 100 (its success at n = 60 was a lucky static ordering).
- Smallest number of slots: for 6 of 8 sizes it equals the clique lower bound and is provably optimal; for n = 60 and 80 the solver found k = ω + 1 (the run with k = ω hit the budget, so optimality is not proven).

---

## 7. Comparison of the algorithms

| Aspect | A\* | Genetic Algorithm | Forward Checking (+ MRV) |
|---|---|---|---|
| Problem type | Shortest path (search) | Tour optimisation (meta-heuristic) | Constraint satisfaction |
| Family | Informed best-first search | Population-based metaheuristic | Systematic search with propagation |
| Complete | Yes | No | Yes |
| Optimal | Yes (consistent heuristic) | No guarantee | n/a (feasibility) |
| Accuracy here | 100 % optimal paths | 99.77 % of exact optimum | 100 % constraint satisfaction up to n = 100 |
| Typical cost here | about 181 nodes, 1.7 ms per query | 60,000–180,000 evaluations, 1.6–5 s | about 100 assignments, about 4 ms |
| Main cost driver | Quality of the heuristic | Population size × generations | Variable ordering and pruning |
| Scalability | Stays fast on the 3,147-node graph | Time grows with K and generations | Almost linear with MRV; collapses without it |
| Reliability across runs | Deterministic | Stochastic, results vary | Deterministic |
| Main weakness | Frontier memory grows | No optimality proof | Static ordering can thrash (FC without MRV) |

These algorithms solve **different problems**, so they cannot be ranked by one number. They can be compared on guarantee, speed and robustness: A\* combines an optimality guarantee with the smallest cost per query; Forward Checking + MRV is also exact and fast but only for feasibility problems; the Genetic Algorithm is the only one that gives up the guarantee, in exchange for flexibility on problems too large for exact methods.

---

## 8. Which algorithm is best? Final recommendation

### Best algorithm overall for the flight-route problem: **A\* search**

It is the only method here that answers the core passenger question, "what is the shortest way from A to B?", with a proof of optimality that was confirmed empirically (100 % of 300 queries, 0 heuristic violations on 184,530 edge checks). It is also the cheapest per query: 181 expanded airports, 1.75 ms and 90 KiB, against 1,519 airports, 4.58 ms and 323 KiB for Dijkstra.

### Recommendation by use case

| Use case | Recommended | Why | Notes |
|---|---|---|---|
| Shortest route between two airports | **A\*** (w = 1) | Optimal, fastest exact method, deterministic | Haversine heuristic; do not use BFS (18.5 % longer paths) |
| Very fast approximate routing | Weighted A\* (w = 1.5) | About 55 expanded nodes | Accept 2.7 % extra distance |
| Multi-city tour up to about 16 cities | Held–Karp (exact) or GA | Exact still feasible; GA reaches 99.77 % | Use the exact result to validate the GA |
| Multi-city tour, 20+ cities | **Genetic Algorithm + local search** | Beats NN + 2-opt by 5.1 % at 30 cities | Population about 200, mutation 0.25–0.5, crossover 0.9, elitism 2; run about 10 seeds, keep the best, polish with 2-opt |
| Slot / resource assignment | **Forward Checking + MRV** | About n assignments, 0 violations | Always use MRV with degree tie-break; start from the DSATUR bound |

### Practical advice

- Do not use plain backtracking or plain forward checking on dense hub networks.
- Never report a GA result from a single run; repeat runs and keep the best.
- Validate every meta-heuristic against an exact method on a small instance first.

### What we learned

1. A heuristic that is both admissible and consistent gives large speed-ups without losing optimality.
2. Meta-heuristics have no optimality guarantee, so they must be validated; hyper-parameters such as the mutation rate visibly change reliability.
3. For constraint satisfaction, propagation and variable-ordering heuristics matter more than raw search speed.
4. Careful data cleaning and extraction is a prerequisite for every experiment.

---

## 9. Tools and technologies

| Tool | Purpose |
|---|---|
| Python 3 | Programming language (all three algorithms implemented from scratch) |
| NumPy, pandas | Arrays, data loading and cleaning |
| SciPy | Reference shortest-path matrices (Dijkstra) for the GA cost matrix |
| NetworkX | Connected components, cliques, DSATUR reference colouring |
| Matplotlib | Charts and maps |
| Jupyter / Google Colab | Executable record of all experiments |

---

## 10. Repository structure

```
.
├── AI_Project_Flight_Routes.ipynb   # full, runnable code and experiments
├── README.md                        # this file
└── (optional) Final_AI_Project_Report.pdf   # full written project report
```

`routes.dat` and `airports.dat` are downloaded automatically by the notebook.

---

## 11. Limitations and future work

**Limitations**

- Costs are distances only (no prices, schedules or layover times).
- The slot-assignment model is a stylised constraint problem.
- The data is a static snapshot of the OpenFlights database.
- Timing figures depend on the machine; the GA hyper-parameter grid uses only five seeds per setting.

**Future work**

- Add layover time, ticket price and airline alliances to the cost function (multi-objective search).
- Use arc consistency (MAC) instead of forward checking.
- Try memetic algorithms (GA + local search) for larger tours.

---

## 12. References

1. OpenFlights. Airport, airline and route databases. <https://openflights.org/data.html>
2. Kaggle. Flight Route Database (open-flights). <https://www.kaggle.com/datasets/openflights/flight-route-database>
3. Hart, P. E., Nilsson, N. J. and Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Trans. Systems Science and Cybernetics*, 4(2), 100–107.
4. Russell, S. and Norvig, P. *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
5. Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
6. Davis, L. (1985). Applying adaptive algorithms to epistatic domains. *Proc. IJCAI*, 162–164. (Order crossover)
7. Haralick, R. M. and Elliott, G. L. (1980). Increasing tree search efficiency for constraint satisfaction problems. *Artificial Intelligence*, 14(3), 263–313. (Forward checking)
8. Held, M. and Karp, R. M. (1962). A dynamic programming approach to sequencing problems. *J. SIAM*, 10(1), 196–210.
9. Sinnott, R. W. (1984). Virtues of the haversine. *Sky and Telescope*, 68(2), 159.
