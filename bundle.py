# Mini bundle method (one-dimensional)
# Finds the minimum of any nonsmooth convex function.
# The function is supplied from OUTSIDE: f and subgradient are passed as parameters.
# This file contains no problem, only the algorithm.


# ─────────────────────────────────────────────
# 1) Cuts and the model
# ─────────────────────────────────────────────

def make_cut(xi, f, subgradient):
    # Evaluate at x = xi and pack three numbers: (point, value, slope)
    return (xi, f(xi), subgradient(xi))


def cut_value(cut, x):
    # Height of the cut's line at x: value + slope * (x - point)
    xi, fi, gi = cut
    return fi + gi * (x - xi)


def model(bundle, x):
    # The maximum height of all cuts at x.
    # An (optimistic) approximation that always stays below the true f.
    return max(cut_value(c, x) for c in bundle)


def linearization_error(cut, center, f):
    # How far the cut lies below the true f at the center (always >= 0)
    # 0 = fresh cut, large = stale cut
    return f(center) - cut_value(cut, center)


# ─────────────────────────────────────────────
# 2) Master problem: "where should I go next?"
#    minimize: model(x) + (x - center)^2 / (2t)
#    Note: this part NEVER sees the true f, only the cuts in the bundle.
# ─────────────────────────────────────────────

def next_point(bundle, center, t):
    # Exact solver: list every candidate where the optimum can be, pick the best.

    # Scoring function: model + proximal term (the "leash")
    def objective(x):
        return model(bundle, x) + (x - center) ** 2 / (2 * t)

    candidates = []

    # "If only this cut were active" -> center - t * slope
    for cut in bundle:
        xi, fi, gi = cut
        candidates.append(center - t * gi)

    # "If two cuts were active together" -> intersection of the two lines
    for i in range(len(bundle)):
        for j in range(i + 1, len(bundle)):
            point1, value1, slope1 = bundle[i]
            point2, value2, slope2 = bundle[j]
            if slope1 == slope2:
                continue   # parallel lines never intersect
            intercept1 = value1 - slope1 * point1   # line: slope1 * x + intercept1
            intercept2 = value2 - slope2 * point2   # line: slope2 * x + intercept2
            candidates.append((intercept2 - intercept1) / (slope1 - slope2))

    # Pick the candidate with the lowest score
    best = candidates[0]
    for x in candidates:
        if objective(x) < objective(best):
            best = x
    return best


def next_point_grid(bundle, center, t):
    # REFEREE: solves the same problem by scanning center ± 10 in steps of 0.01.
    # Slow but fully transparent; used only in tests.
    best_x = center
    lowest = model(bundle, center)
    x = center - 10
    while x <= center + 10:
        score = model(bundle, x) + (x - center) ** 2 / (2 * t)
        if score < lowest:
            lowest = score
            best_x = x
        x = x + 0.01
    return best_x


# ─────────────────────────────────────────────
# 3) Main algorithm
# ─────────────────────────────────────────────

def run_bundle(f, subgradient, start_center, start_t, max_iter, verbose=True):
    # f, subgradient: the function to minimize and its slope (from outside)
    # Returns: the list of centers per iteration (last element = minimum found)
    center = start_center
    t = start_t
    bundle = [make_cut(center, f, subgradient)]   # first evaluation
    history = [center]

    for i in range(max_iter):
        # CANDIDATE: propose the next point using the model and the leash
        x = next_point(bundle, center, t)

        # STOP?: if the model (even though optimistic) promises no improvement, stop
        predicted_decrease = f(center) - model(bundle, x)
        if predicted_decrease < 0.000001:
            if verbose:
                print(f"--> Stopped at iteration {i}, center: {center}")
            break

        # EVALUATE: go to the candidate, measure true value and slope, add to the bundle
        bundle.append(make_cut(x, f, subgradient))

        # DECISION: is the candidate really lower?
        if f(x) < f(center):
            # Serious step: move, the model was reliable -> lengthen the leash
            center = x
            t = t * 2
            decision = "SERIOUS (moved)"
        else:
            # Null step: stay, the model was wrong -> shorten the leash
            t = t / 2
            decision = "NULL (stayed)"

        history.append(center)
        if verbose:
            print(f"Iter {i:2d} | x: {x:.4f} | t: {t} | {decision}")

    return history
