import random
import torch
from torch import nn
from bundle import kesi_yap, yeni_nokta_aktif, model as harita, dogrusallastirma_hatasi
from veri_topla import en_iyi_t_bul, T_ADAYLARI
from problemler import knapsack_uret, rastgele_knapsack
from lstm_egit import TModeli


def ag_yukle(dosya_adi):
    # Build the same network shape and load the trained knobs into it
    ag = nn.Sequential(
        nn.Linear(5, 32),
        nn.ReLU(),
        nn.Linear(32, len(T_ADAYLARI)),
    )
    ag.load_state_dict(torch.load(dosya_adi))
    return ag


def lstm_yukle(dosya_adi):
    lstm = TModeli()
    lstm.load_state_dict(torch.load(dosya_adi))
    return lstm


def ozellikler(t, son_serious, demet, merkez, f):
    # The same 5 features as in kayitli_calistir, scaled the same way
    x = yeni_nokta_aktif(demet, merkez, t)
    vaat = f(merkez) - harita(demet, x)
    en_buyuk_hata = max(dogrusallastirma_hatasi(k, merkez, f) for k in demet)
    girdi = torch.tensor([[t, vaat, son_serious, len(demet), en_buyuk_hata]])
    return torch.log1p(girdi)


def tur_say(f, subgradient, baslangic_merkez, baslangic_t, yontem, ag=None, lstm=None, max_tur=100):
    # Run bundle with the chosen t rule and return the number of iterations
    merkez = baslangic_merkez
    t = baslangic_t
    demet = [kesi_yap(merkez, f, subgradient)]
    son_serious = 0
    durum = None   # LSTM memory (h, c); empty at the start of every problem

    for i in range(max_tur):
        if yontem == "kural":
            t_kullan = t
        elif yontem == "ogretmen":
            t_kullan = en_iyi_t_bul(demet, merkez, f)
        elif yontem == "mlp":
            with torch.no_grad():
                puanlar = ag(ozellikler(t, son_serious, demet, merkez, f))
            t_kullan = T_ADAYLARI[puanlar.argmax().item()]
        else:  # "lstm": one step at a time, memory carried to the next iteration
            girdi = ozellikler(t, son_serious, demet, merkez, f).unsqueeze(0)
            with torch.no_grad():
                puanlar, durum = lstm(girdi, durum)
            t_kullan = T_ADAYLARI[puanlar.argmax().item()]

        x = yeni_nokta_aktif(demet, merkez, t_kullan)
        if f(merkez) - harita(demet, x) < 0.000001:
            return i

        demet.append(kesi_yap(x, f, subgradient))
        if f(x) < f(merkez):
            merkez = x
            t = t_kullan * 2
            son_serious = 1
        else:
            t = t_kullan / 2
            son_serious = 0

    return max_tur


if __name__ == "__main__":
    ag = ag_yukle("model.pt")
    lstm = lstm_yukle("lstm.pt")
    random.seed(123)
    sonuclar = {"kural": [], "ogretmen": [], "mlp": [], "lstm": []}

    for p in range(300):
        degerler, agirliklar, kapasite = rastgele_knapsack()
        f, subgradient = knapsack_uret(degerler, agirliklar, kapasite)
        baslangic = random.uniform(0, 5)
        baslangic_t = random.choice([0.1, 0.5, 1, 2, 5])
        for yontem in sonuclar:
            sonuclar[yontem].append(tur_say(f, subgradient, baslangic, baslangic_t, yontem, ag, lstm))

    print("300 new problems, average number of iterations:")
    for yontem, turlar in sonuclar.items():
        print(f"  {yontem:9s}: {sum(turlar) / len(turlar):.2f}   (worst: {max(turlar)})")
