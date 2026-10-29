# On the knapsack dual: subgradient step rules (Project 1) vs the bundle method
import matplotlib.pyplot as plt
from bundle import run_bundle
from knapsack import f, subgradient

F_STAR = 14   # minimum value of the dual (Polyak needs to know it)


# ─── Step rules and subgradient loop ───
def step_constant(x):
    return 0.2


def step_diminishing(k):
    return 0.2 / (k + 1)


def step_polyak(x, g, fstar):
    if g == 0:
        return 0
    return (f(x) - fstar) / (g ** 2)


def run_subgradient(rule, x0, num_steps):
    x = x0
    history = [x]
    for k in range(num_steps):
        g = subgradient(x)
        if rule == "constant":
            alpha = step_constant(x)
        elif rule == "diminishing":
            alpha = step_diminishing(k)
        elif rule == "polyak":
            alpha = step_polyak(x, g, F_STAR)
        x = x - alpha * g
        history.append(x)
    return history


# ─── Comparison ───
if __name__ == "__main__":
    x0 = 3
    steps = 30

    plt.plot(run_subgradient("constant", x0, steps), label="subgradient: constant")
    plt.plot(run_subgradient("diminishing", x0, steps), label="subgradient: diminishing")
    plt.plot(run_subgradient("polyak", x0, steps), label="subgradient: Polyak (f* known)")
    plt.plot(run_bundle(f, subgradient, x0, 0.1, steps, verbose=False),
             label="bundle", linewidth=3)

    plt.axhline(2, color="gray", linestyle="--")
    plt.xlabel("iteration")
    plt.ylabel("λ (price per unit of capacity)")
    plt.title("Knapsack dual: λ over iterations")
    plt.legend()
    plt.savefig("comparison.png")
    plt.show()
