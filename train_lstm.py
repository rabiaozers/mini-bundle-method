# LSTM that predicts the best t, reading a problem's iterations in order
import csv
import random
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from collect_data import T_CANDIDATES
from train_mlp import FEATURES

PAD = -100   # label used for padded positions; the loss ignores it


def read_sequences(file_name):
    # Read data.csv and group the rows by problem: one sequence per problem
    groups = {}
    with open(file_name, encoding="utf-8") as file:
        for row in csv.DictReader(file):
            p = row["problem"]
            if p in groups:
                groups[p].append(row)
            else:
                groups[p] = [row]

    sequences = []
    for p in groups:
        X = torch.tensor([[float(r[name]) for name in FEATURES] for r in groups[p]])
        y = torch.tensor([T_CANDIDATES.index(float(r["best_t"])) for r in groups[p]])
        sequences.append((torch.log1p(X), y))
    return sequences


class TModel(nn.Module):
    # LSTM (memory over iterations) + a linear head (one score per candidate t)
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(len(FEATURES), hidden, batch_first=True)
        self.head = nn.Linear(hidden, len(T_CANDIDATES))

    def forward(self, x, state=None):
        # x: (problems, iterations, features), state: memory (h, c) or None
        out, state = self.lstm(x, state)
        return self.head(out), state


def pad_batch(sequences):
    # Pad sequences of different lengths to one rectangular batch
    X = pad_sequence([x for x, _ in sequences], batch_first=True)
    y = pad_sequence([y for _, y in sequences], batch_first=True, padding_value=PAD)
    return X, y


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    # 1) Data: split by PROBLEM, not by row (no leakage)
    sequences = read_sequences("data.csv")
    random.shuffle(sequences)
    split = int(len(sequences) * 0.8)
    X_train, y_train = pad_batch(sequences[:split])
    X_test, y_test = pad_batch(sequences[split:])
    print("training problems:", len(sequences[:split]), "| test problems:", len(sequences[split:]))

    # 2) Model, loss, optimizer
    model = TModel()
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 3) Training loop
    for epoch in range(500):
        pred, _ = model(X_train)
        loss = loss_fn(pred.reshape(-1, len(T_CANDIDATES)), y_train.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"epoch {epoch:3d} | loss: {loss.item():.3f}")

    # 4) Evaluation (only real iterations, padding excluded)
    with torch.no_grad():
        pred, _ = model(X_test)
    pred_class = pred.argmax(dim=2)
    real = y_test != PAD
    accuracy = ((pred_class == y_test) & real).sum().item() / real.sum().item()
    baseline = ((y_test == 0) & real).sum().item() / real.sum().item()
    print(f"baseline (always 0.1): {baseline:.1%}")
    print(f"LSTM accuracy:         {accuracy:.1%}")

    torch.save(model.state_dict(), "lstm.pt")
    print("model saved -> lstm.pt")
