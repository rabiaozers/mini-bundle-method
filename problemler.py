import random
import csv
from veri_topla import kayitli_calistir


def knapsack_uret(degerler, agirliklar, kapasite):
    # Builds and returns the f and subgradient functions for the given items

    def f(x):
        toplam = 0
        for i in range(len(degerler)):
            toplam = toplam + max(0, degerler[i] - agirliklar[i] * x)
        return toplam + kapasite * x

    def subgradient(x):
        kullanilan = 0
        for i in range(len(degerler)):
            if degerler[i] - agirliklar[i] * x > 0:
                kullanilan = kullanilan + agirliklar[i]
        return kapasite - kullanilan

    return f, subgradient


def rastgele_knapsack():
    # A random knapsack problem: 3-10 items, random values/weights/capacity
    n = random.randint(3, 10)
    degerler = [random.randint(1, 20) for _ in range(n)]
    agirliklar = [random.randint(1, 10) for _ in range(n)]
    kapasite = random.randint(1, sum(agirliklar))
    return degerler, agirliklar, kapasite

def veri_seti_olustur(problem_sayisi):
    # Solves many random problems and collects all training rows
    tum_satirlar = []
    for p in range(problem_sayisi):
        degerler, agirliklar, kapasite = rastgele_knapsack()
        f, subgradient = knapsack_uret(degerler, agirliklar, kapasite)
        baslangic = random.uniform(0, 5)
        baslangic_t = random.choice([0.1, 0.5, 1, 2, 5])
        satirlar = kayitli_calistir(f, subgradient, baslangic, baslangic_t, 30)
        for satir in satirlar :
            satir['problem'] = p
        tum_satirlar.extend(satirlar)
    return tum_satirlar


def csv_kaydet(satirlar, dosya_adi):
    # Writes the rows to a table file that can also be opened in Excel
    with open(dosya_adi, "w", newline="", encoding="utf-8") as dosya:
        yazici = csv.DictWriter(dosya, fieldnames=satirlar[0].keys())
        yazici.writeheader()
        yazici.writerows(satirlar)

if __name__ == "__main__":
    random.seed(42)
    satirlar = veri_seti_olustur(1000)
    csv_kaydet(satirlar, "veri.csv")
    print(len(satirlar), "satır kaydedildi -> veri.csv")