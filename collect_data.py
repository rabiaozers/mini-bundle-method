# Training data for learned t: at every bundle iteration, record the state
# and the t that a greedy teacher would choose.
from bundle import make_cut, next_point, model, linearization_error
import knapsack

T_CANDIDATES = [0.1, 0.2, 0.5, 1, 2, 5, 10]


def improvement(bundle, center, t, f):
    # How much would f decrease if we took one step with this t?
    x = next_point(bundle, center, t)
    return f(center) - f(x)


def best_t(bundle, center, f):
    # Greedy teacher: try every candidate t and pick the one with the largest improvement
    best = T_CANDIDATES[0]
    for t in T_CANDIDATES:
        if improvement(bundle, center, t, f) > improvement(bundle, center, best, f):
            best = t
    return best


def run_and_record(f, subgradient, start_center, start_t, max_iter):
    # Same as run_bundle, but records one training row per iteration
    center = start_center
    t = start_t
    bundle = [make_cut(center, f, subgradient)]
    last_serious = 0
    rows = []

    for i in range(max_iter):
        x = next_point(bundle, center, t)
        predicted_decrease = f(center) - model(bundle, x)
        if predicted_decrease < 0.000001:
            break

        # Measure the state and record it
        errors = [linearization_error(c, center, f) for c in bundle]
        rows.append({
            "t": t,
            "predicted_decrease": predicted_decrease,
            "last_serious": last_serious,
            "num_cuts": len(bundle),
            "max_lin_error": max(errors),
            "best_t": best_t(bundle, center, f),
        })

        bundle.append(make_cut(x, f, subgradient))
        if f(x) < f(center):
            center = x
            t = t * 2
            last_serious = 1
        else:
            t = t / 2
            last_serious = 0

    return rows


if __name__ == "__main__":
    for row in run_and_record(knapsack.f, knapsack.subgradient, 3, 0.1, 30):
        print(row)
