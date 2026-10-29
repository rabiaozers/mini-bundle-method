# Learning the bundle step parameter t

The bundle method in this repo updates its proximal parameter `t` with a fixed
rule: double it after a serious step, halve it after a null step. In my Erasmus+
traineeship I worked on BundleSolverML, which replaces such a rule inside the
SMS++ solver with an LSTM that predicts `t`. This experiment is a small, fully
transparent Python version of that idea: can a network learn a better `t` than
the rule?

## Setup

**Problems.** 1000 random knapsack duals (3 to 10 items, random values,
weights and capacity), each started from a random point with a random initial `t`.

**Labels: a greedy teacher.** At every iteration of every run, I freeze the
state, try 7 candidate values `t ∈ {0.1, 0.2, 0.5, 1, 2, 5, 10}`, take one step
with each, and record the `t` that decreases L the most. This gives 3746
(state, best t) pairs from 968 problems (32 problems start at the optimum).

**Features (5 per iteration):** current `t`, predicted decrease of the model,
whether the last step was serious, number of cuts, and the largest
linearization error in the bundle. All scaled with `log1p`.

**Models** (PyTorch), both trained as 7-way classifiers:

- MLP `5 → 32 → 7`: sees each iteration on its own.
- LSTM (hidden size 32) + linear head: reads a problem's iterations in order
  and carries its memory `(h, c)` from one iteration to the next. Train/test
  split by problem, so no run is split between the two sets.

## Results

Classification accuracy on held-out data:

| model | accuracy | baseline (always predict 0.1) |
|---|---|---|
| MLP | 63.7% | 62.4% |
| LSTM | 68.6% | 64.4% |

What actually matters is speed. Average number of bundle iterations on 300 new
random problems:

| how `t` is chosen | average | worst |
|---|---|---|
| rule (×2 / ÷2) | 3.82 | 8 |
| greedy teacher (tries 7 values every iteration) | 3.31 | 7 |
| MLP | 5.06 | 44 |
| LSTM | 4.63 | 15 |

## What I learned

1. **Better accuracy did not mean a faster solver.** Both networks beat the
   accuracy baseline, yet both made the bundle method slower than the simple
   rule. The model metric and the real metric are different things.
2. **Imbalanced labels.** 63% of the labels are `0.1`, so the cheapest way to
   lower the loss is to predict `0.1` almost always. The MLP did exactly that
   (694 of 750 test predictions), which turns the bundle method into a tiny
   constant step; in the worst problem it needed 44 iterations instead of 7.
3. **Memory helps.** The rule implicitly remembers history by letting `t`
   accumulate. The MLP cannot; the LSTM can, and its worst case dropped from 44
   to 15 iterations.
4. **The ceiling was low from the start.** Even the teacher, which tries every
   candidate at every step, saves only half an iteration over the rule. On a
   one-dimensional dual the bundle method finishes in a few iterations and the
   rule is already close to the best greedy choice. Learned `t` only has room
   to pay off on large, high-dimensional duals with hundreds of iterations,
   which is the setting of SMS++.

## Files

- `veri_topla.py`: greedy teacher (`en_iyi_t_bul`) and bundle run that records one row per iteration
- `problemler.py`: random knapsack generator, writes `veri.csv`
- `egit.py`: MLP training (`model.pt`)
- `lstm_egit.py`: LSTM training on per-problem sequences (`lstm.pt`)
- `t_karsilastir.py`: rule vs teacher vs MLP vs LSTM inside the bundle method

## Run

```
pip install torch
python problemler.py
python egit.py
python lstm_egit.py
python t_karsilastir.py
```
