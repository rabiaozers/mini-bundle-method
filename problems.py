# Random knapsack duals and the training dataset (data.csv)
import random
import csv
from collect_data import run_and_record


def make_knapsack(values, weights, capacity):
    # Builds and returns the f and subgradient functions for the given items

    def f(x):
        total = 0
        for i in range(len(values)):
            total = total + max(0, values[i] - weights[i] * x)
        return total + capacity * x

    def subgradient(x):
        used = 0
        for i in range(len(values)):
            if values[i] - weights[i] * x > 0:
                used = used + weights[i]
        return capacity - used

    return f, subgradient


def random_knapsack():
    # A random knapsack problem: 3-10 items, random values/weights/capacity
    n = random.randint(3, 10)
    values = [random.randint(1, 20) for _ in range(n)]
    weights = [random.randint(1, 10) for _ in range(n)]
    capacity = random.randint(1, sum(weights))
    return values, weights, capacity


def build_dataset(num_problems):
    # Solves many random problems and collects all training rows
    all_rows = []
    for p in range(num_problems):
        values, weights, capacity = random_knapsack()
        f, subgradient = make_knapsack(values, weights, capacity)
        start = random.uniform(0, 5)
        start_t = random.choice([0.1, 0.5, 1, 2, 5])
        rows = run_and_record(f, subgradient, start, start_t, 30)
        for row in rows:
            row["problem"] = p
        all_rows.extend(rows)
    return all_rows


def save_csv(rows, file_name):
    # Writes the rows to a table file that can also be opened in Excel
    with open(file_name, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    random.seed(42)
    rows = build_dataset(1000)
    save_csv(rows, "data.csv")
    print(len(rows), "rows saved -> data.csv")
