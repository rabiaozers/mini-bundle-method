from bundle import bundle_calistir, kesi_yap, yeni_nokta_grid, yeni_nokta_aktif
import knapsack


# ─── Test için oyuncak fonksiyon: köşe x = 1'de, minimum değeri -1 ───
def oyuncak_f(x):
    return max(-2 * x + 1, x - 2)

def oyuncak_subgradient(x):
    if x <= 1:
        return -2
    else:
        return 1


def test_kink_minimum_bulma():
    # Algoritma oyuncak fonksiyonun minimumunu (x = 1) bulmalı
    gecmis = bundle_calistir(oyuncak_f, oyuncak_subgradient, 3, 0.1, 30, yazdir=False)
    assert abs(gecmis[-1] - 1) < 0.001


def test_tek_kesi_subgradient_adimi():
    # Tek kesiyle aday, Proje 1'deki adımla aynı olmalı: merkez - t * eğim
    demet = [kesi_yap(3, oyuncak_f, oyuncak_subgradient)]   # eğim 1
    assert yeni_nokta_aktif(demet, 3, 1) == 2


def test_grid_aktif():
    # Hakem testi: kendi çözücümüz, grid ile aynı aday noktayı bulmalı
    demet = [kesi_yap(x, oyuncak_f, oyuncak_subgradient) for x in [3, -1, 0.5]]
    fark = yeni_nokta_grid(demet, 0.5, 1) - yeni_nokta_aktif(demet, 0.5, 1)
    assert abs(fark) < 0.01


def test_knapsack_dual_kagit_degerleri():
    # Kâğıtta elle hesapladığımız değerler
    assert knapsack.f(0) == 20
    assert knapsack.f(1) == 15
    assert knapsack.f(2) == 14
    assert knapsack.subgradient(1.5) == -1
    assert knapsack.subgradient(2.5) == 4


def test_knapsack_dual_minimumu():
    # Bundle, dual'ın dibini λ = 2'de bulmalı; üst sınır 14 ve gerçek değerin (10) üstünde
    gecmis = bundle_calistir(knapsack.f, knapsack.subgradient, 3, 0.1, 30, yazdir=False)
    lam = gecmis[-1]
    assert abs(lam - 2) < 0.001
    assert abs(knapsack.f(lam) - 14) < 0.001
    assert knapsack.f(lam) >= 10   # zayıf dualite