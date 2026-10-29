# Lagrangian dual of a small knapsack problem.
# Dualizing the capacity constraint with a price λ gives
#   L(λ) = Σ max(0, value_i - λ·weight_i) + λ·capacity
# Every L(λ) is an upper bound on the true optimum (weak duality);
# the bundle method finds the tightest one by minimizing L.

from bundle import run_bundle

values = [6, 10, 4]
weights = [2, 5, 4]
capacity = 6


def f(x):
    # Dual function L(λ): each item's net profit (if positive) + capacity rent
    total = 0
    for i in range(len(values)):
        total = total + max(0, values[i] - weights[i] * x)
    total = total + capacity * x
    return total


def subgradient(x):
    # Slope = capacity - (total weight of items with positive net profit)
    # Negative: knapsack overflows, price should rise. Positive: space left, price should fall.
    used = 0
    for i in range(len(values)):
        if values[i] - weights[i] * x > 0:
            used = used + weights[i]
    return capacity - used


if __name__ == "__main__":
    print("Knapsack dual solved with the bundle method")
    print("-" * 45)
    history = run_bundle(f, subgradient, 3, 0.1, 30)
    lam = history[-1]
    print("-" * 45)
    print(f"Best price λ = {lam}")
    print(f"Upper bound L(λ) = {f(lam)}   (true optimum: 10)")
