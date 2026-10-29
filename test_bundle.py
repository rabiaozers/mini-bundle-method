from bundle import run_bundle, make_cut, next_point_grid, next_point, linearization_error
import knapsack


# ─── Toy function for tests: kink at x = 1, minimum value -1 ───
def toy_f(x):
    return max(-2 * x + 1, x - 2)


def toy_subgradient(x):
    if x <= 1:
        return -2
    else:
        return 1


def test_finds_kink_minimum():
    # The algorithm should find the toy function's minimum (x = 1)
    history = run_bundle(toy_f, toy_subgradient, 3, 0.1, 30, verbose=False)
    assert abs(history[-1] - 1) < 0.001


def test_single_cut_is_subgradient_step():
    # With a single cut, the candidate equals a subgradient step: center - t * slope
    bundle = [make_cut(3, toy_f, toy_subgradient)]   # slope 1
    assert next_point(bundle, 3, 1) == 2


def test_exact_solver_matches_grid():
    # Referee test: our solver should find the same candidate as the grid search
    bundle = [make_cut(x, toy_f, toy_subgradient) for x in [3, -1, 0.5]]
    diff = next_point_grid(bundle, 0.5, 1) - next_point(bundle, 0.5, 1)
    assert abs(diff) < 0.01


def test_knapsack_dual_hand_values():
    # Values computed by hand on paper
    assert knapsack.f(0) == 20
    assert knapsack.f(1) == 15
    assert knapsack.f(2) == 14
    assert knapsack.subgradient(1.5) == -1
    assert knapsack.subgradient(2.5) == 4


def test_knapsack_dual_minimum():
    # Bundle should find the dual's minimum at λ = 2; upper bound 14, above the true optimum (10)
    history = run_bundle(knapsack.f, knapsack.subgradient, 3, 0.1, 30, verbose=False)
    lam = history[-1]
    assert abs(lam - 2) < 0.001
    assert abs(knapsack.f(lam) - 14) < 0.001
    assert knapsack.f(lam) >= 10   # weak duality


def test_linearization_error():
    # A cut is exact at its own point and stale away from it
    cut = make_cut(3, knapsack.f, knapsack.subgradient)
    assert linearization_error(cut, 3, knapsack.f) == 0
    assert linearization_error(cut, 2, knapsack.f) == 2
