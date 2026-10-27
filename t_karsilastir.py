import random
import torch
from torch import nn
from bundle import kesi_yap, yeni_nokta_aktif, model as harita, dogrusallastirma_hatasi
from veri_topla import en_iyi_t_bul, T_ADAYLARI
from problemler import knapsack_uret, rastgele_knapsack


def ag_yukle(dosya_adi):
    # Build the same network shape and load the trained knobs into it
    ag = nn.Sequential(
        nn.Linear(5, 32),
        nn.ReLU(),
        nn.Linear(32, len(T_ADAYLARI)),
    )
    ag.load_state_dict(torch.load(dosya_adi))
    return ag


def ag_t_sec(ag, t, son_serious, demet, merkez, f):
    # Compute the same 5 features as in kayitli_calistir and ask the network
    x = yeni_nokta_aktif(demet, merkez, t)
    vaat = f(merkez) - harita(demet, x)
    en_buyuk_hata = max(dogrusallastirma_hatasi(k, merkez, f) for k in demet)
    girdi = torch.tensor([[t, vaat, son_serious, len(demet), en_buyuk_hata]])
    girdi = torch.log1p(girdi)
    sira = ag(girdi).argmax(dim=1).item()
    return T_ADAYLARI[sira]


def tur_say(f, subgradient, baslangic_merkez, baslangic_t, yontem, ag=None, max_tur=100):
    # Run bundle with the chosen t rule and return the number of iterations
    merkez = baslangic_merkez
    t = baslangic_t
    demet = [kesi_yap(merkez, f, subgradient)]
    son_serious = 0

    for i in range(max_tur):
        if yontem == "kural":
            t_kullan = t
        elif yontem == "ogretmen":
            t_kullan = en_iyi_t_bul(demet, merkez, f)
        else:
            t_kullan = ag_t_sec(ag, t, son_serious, demet, merkez, f)

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
    random.seed(123)
    sonuclar = {"kural": [], "ogretmen": [], "ag": []}

    for p in range(300):
        degerler, agirliklar, kapasite = rastgele_knapsack()
        f, subgradient = knapsack_uret(degerler, agirliklar, kapasite)
        baslangic = random.uniform(0, 5)
        baslangic_t = random.choice([0.1, 0.5, 1, 2, 5])
        for yontem in sonuclar:
            sonuclar[yontem].append(tur_say(f, subgradient, baslangic, baslangic_t, yontem, ag))

    print("300 yeni problem, ortalama tur sayısı:")
    for yontem, turlar in sonuclar.items():
        print(f"  {yontem:9s}: {sum(turlar) / len(turlar):.2f}   (en kötü: {max(turlar)})")