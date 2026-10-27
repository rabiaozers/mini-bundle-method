import csv
import torch
from torch import nn
from veri_topla import T_ADAYLARI

OZELLIKLER = ["t", "vaat", "son_serious", "kesi_sayisi", "en_buyuk_hata"]


def veriyi_oku(dosya_adi):
    # Read veri.csv: features -> X, index of the best t -> y
    X = []
    y = []
    with open(dosya_adi, encoding="utf-8") as dosya:
        for satir in csv.DictReader(dosya):
            X.append([float(satir[ad]) for ad in OZELLIKLER])
            y.append(T_ADAYLARI.index(float(satir["en_iyi_t"])))
    return torch.tensor(X), torch.tensor(y)


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) Data
    X, y = veriyi_oku("veri.csv")
    X = torch.log1p(X)

    sira = torch.randperm(len(X))
    sinir = int(len(X) * 0.8)
    X_egitim, y_egitim = X[sira[:sinir]], y[sira[:sinir]]
    X_test, y_test = X[sira[sinir:]], y[sira[sinir:]]

    # 2) Model
    model = nn.Sequential(
        nn.Linear(5, 32),
        nn.ReLU(),
        nn.Linear(32, len(T_ADAYLARI)),
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 3) Training loop
    for tur in range(500):
        tahmin = model(X_egitim)
        hata = loss_fn(tahmin, y_egitim)
        optimizer.zero_grad()
        hata.backward()
        optimizer.step()
        if tur % 100 == 0:
            print(f"tur {tur:3d} | hata: {hata.item():.3f}")

    # 4) Evaluation
    tahmin_sinif = model(X_test).argmax(dim=1)
    dogruluk = (tahmin_sinif == y_test).float().mean().item()
    baseline = (y_test == 0).float().mean().item()
    print(f"baseline (always 0.1): {baseline:.1%}")
    print(f"network accuracy:      {dogruluk:.1%}")
    from collections import Counter

    torch.save(model.state_dict(), "model.pt")
    print("model kaydedildi -> model.pt")