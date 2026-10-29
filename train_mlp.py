# MLP that predicts the best t from the current state of a bundle iteration
import csv
from collections import Counter
import torch
from torch import nn
from collect_data import T_CANDIDATES

FEATURES = ["t", "predicted_decrease", "last_serious", "num_cuts", "max_lin_error"]


def read_data(file_name):
    # Read data.csv: features -> X, index of the best t -> y
    X = []
    y = []
    with open(file_name, encoding="utf-8") as file:
        for row in csv.DictReader(file):
            X.append([float(row[name]) for name in FEATURES])
            y.append(T_CANDIDATES.index(float(row["best_t"])))
    return torch.tensor(X), torch.tensor(y)


def make_mlp():
    return nn.Sequential(
        nn.Linear(len(FEATURES), 32),
        nn.ReLU(),
        nn.Linear(32, len(T_CANDIDATES)),
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) Data: scale, shuffle, split 80/20
    X, y = read_data("data.csv")
    X = torch.log1p(X)

    order = torch.randperm(len(X))
    split = int(len(X) * 0.8)
    X_train, y_train = X[order[:split]], y[order[:split]]
    X_test, y_test = X[order[split:]], y[order[split:]]

    # 2) Model, loss, optimizer
    model = make_mlp()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 3) Training loop
    for epoch in range(500):
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"epoch {epoch:3d} | loss: {loss.item():.3f}")

    # 4) Evaluation
    with torch.no_grad():
        pred_class = model(X_test).argmax(dim=1)
    accuracy = (pred_class == y_test).float().mean().item()
    baseline = (y_test == 0).float().mean().item()
    print(f"baseline (always 0.1): {baseline:.1%}")
    print(f"MLP accuracy:          {accuracy:.1%}")
    print("predicted classes:", Counter(pred_class.tolist()))

    torch.save(model.state_dict(), "mlp.pt")
    print("model saved -> mlp.pt")
