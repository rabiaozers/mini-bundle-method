# Inside the bundle method: rule vs greedy teacher vs MLP vs LSTM for choosing t
import random
import torch
import matplotlib.pyplot as plt
from bundle import make_cut, next_point, model as bundle_model, linearization_error
from collect_data import best_t, T_CANDIDATES
from problems import make_knapsack, random_knapsack
from train_mlp import make_mlp
from train_lstm import TModel


def load_mlp(file_name):
    # Build the same network shape and load the trained weights into it
    mlp = make_mlp()
    mlp.load_state_dict(torch.load(file_name))
    return mlp


def load_lstm(file_name):
    lstm = TModel()
    lstm.load_state_dict(torch.load(file_name))
    return lstm


def features(t, last_serious, bundle, center, f):
    # The same 5 features as in run_and_record, scaled the same way
    x = next_point(bundle, center, t)
    predicted_decrease = f(center) - bundle_model(bundle, x)
    max_error = max(linearization_error(c, center, f) for c in bundle)
    row = torch.tensor([[t, predicted_decrease, last_serious, len(bundle), max_error]])
    return torch.log1p(row)


def count_iterations(f, subgradient, start_center, start_t, method, mlp=None, lstm=None, max_iter=100):
    # Run the bundle method with the chosen way of picking t; return the number of iterations
    center = start_center
    t = start_t
    bundle = [make_cut(center, f, subgradient)]
    last_serious = 0
    state = None   # LSTM memory (h, c); empty at the start of every problem

    for i in range(max_iter):
        if method == "rule":
            t_used = t
        elif method == "teacher":
            t_used = best_t(bundle, center, f)
        elif method == "mlp":
            with torch.no_grad():
                scores = mlp(features(t, last_serious, bundle, center, f))
            t_used = T_CANDIDATES[scores.argmax().item()]
        else:  # "lstm": one iteration at a time, memory carried to the next one
            x_in = features(t, last_serious, bundle, center, f).unsqueeze(0)
            with torch.no_grad():
                scores, state = lstm(x_in, state)
            t_used = T_CANDIDATES[scores.argmax().item()]

        x = next_point(bundle, center, t_used)
        if f(center) - bundle_model(bundle, x) < 0.000001:
            return i

        bundle.append(make_cut(x, f, subgradient))
        if f(x) < f(center):
            center = x
            t = t_used * 2
            last_serious = 1
        else:
            t = t_used / 2
            last_serious = 0

    return max_iter


if __name__ == "__main__":
    mlp = load_mlp("mlp.pt")
    lstm = load_lstm("lstm.pt")
    random.seed(123)
    results = {"rule": [], "teacher": [], "mlp": [], "lstm": []}

    for p in range(300):
        values, weights, capacity = random_knapsack()
        f, subgradient = make_knapsack(values, weights, capacity)
        start = random.uniform(0, 5)
        start_t = random.choice([0.1, 0.5, 1, 2, 5])
        for method in results:
            results[method].append(count_iterations(f, subgradient, start, start_t, method, mlp, lstm))

    print("300 new problems, average number of iterations:")
    averages = {}
    for method, iters in results.items():
        averages[method] = sum(iters) / len(iters)
        print(f"  {method:8s}: {averages[method]:.2f}   (worst: {max(iters)})")

    labels = {"rule": "rule (×2 / ÷2)", "teacher": "greedy teacher", "mlp": "MLP", "lstm": "LSTM"}
    plt.figure(figsize=(6, 3.5))
    bars = plt.bar([labels[m] for m in averages], list(averages.values()),
                   color=["#4c72b0", "#8c8c8c", "#dd8452", "#55a868"])
    for bar, value in zip(bars, averages.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.2f}", ha="center")
    plt.ylim(0, max(averages.values()) * 1.15)
    plt.ylabel("average iterations (lower is better)")
    plt.title("Choosing t inside the bundle method (300 random problems)")
    plt.tight_layout()
    plt.savefig("learned_t.png", dpi=150)
    print("chart saved -> learned_t.png")
