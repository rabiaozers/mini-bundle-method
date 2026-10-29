import csv
import random
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from veri_topla import T_ADAYLARI

OZELLIKLER = ["t", "vaat", "son_serious", "kesi_sayisi", "en_buyuk_hata"]
BOS = -100   # label used for padded positions; the loss ignores it


def dizileri_oku(dosya_adi):
    # Read veri.csv and group the rows by problem: one sequence per problem
    gruplar = {}
    with open(dosya_adi, encoding="utf-8") as dosya:
        for satir in csv.DictReader(dosya):
            p = satir["problem"]
            if p in gruplar:
                gruplar[p].append(satir)
            else:
                gruplar[p] = [satir]

    diziler = []
    for p in gruplar:
        X = torch.tensor([[float(s[ad]) for ad in OZELLIKLER] for s in gruplar[p]])
        y = torch.tensor([T_ADAYLARI.index(float(s["en_iyi_t"])) for s in gruplar[p]])
        diziler.append((torch.log1p(X), y))
    return diziler


class TModeli(nn.Module):
    # LSTM (memory over iterations) + a linear head (one score per candidate t)
    def __init__(self, gizli=32):
        super().__init__()
        self.lstm = nn.LSTM(len(OZELLIKLER), gizli, batch_first=True)
        self.cikis = nn.Linear(gizli, len(T_ADAYLARI))

    def forward(self, x, durum=None):
        # x: (problems, iterations, 5), durum: memory (h, c) or None
        cikti, durum = self.lstm(x, durum)
        return self.cikis(cikti), durum


def paketle(diziler):
    # Pad sequences of different lengths to one rectangular batch
    X = pad_sequence([x for x, _ in diziler], batch_first=True)
    y = pad_sequence([y for _, y in diziler], batch_first=True, padding_value=BOS)
    return X, y


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    # 1) Data: split by PROBLEM, not by row (no leakage)
    diziler = dizileri_oku("veri.csv")
    random.shuffle(diziler)
    sinir = int(len(diziler) * 0.8)
    X_egitim, y_egitim = paketle(diziler[:sinir])
    X_test, y_test = paketle(diziler[sinir:])
    print("training problems:", len(diziler[:sinir]), "| test problems:", len(diziler[sinir:]))

    # 2) Model, loss, optimizer
    model = TModeli()
    loss_fn = nn.CrossEntropyLoss(ignore_index=BOS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 3) Training loop
    for tur in range(500):
        tahmin, _ = model(X_egitim)
        hata = loss_fn(tahmin.reshape(-1, len(T_ADAYLARI)), y_egitim.reshape(-1))
        optimizer.zero_grad()
        hata.backward()
        optimizer.step()
        if tur % 100 == 0:
            print(f"tur {tur:3d} | hata: {hata.item():.3f}")

    # 4) Evaluation (only real iterations, padding excluded)
    with torch.no_grad():
        tahmin, _ = model(X_test)
    tahmin_sinif = tahmin.argmax(dim=2)
    gercek = y_test != BOS
    dogruluk = ((tahmin_sinif == y_test) & gercek).sum().item() / gercek.sum().item()
    baseline = ((y_test == 0) & gercek).sum().item() / gercek.sum().item()
    print(f"baseline (always 0.1): {baseline:.1%}")
    print(f"LSTM accuracy:         {dogruluk:.1%}")

    torch.save(model.state_dict(), "lstm.pt")
    print("model kaydedildi -> lstm.pt")
