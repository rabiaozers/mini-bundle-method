from bundle import bundle_calistir, kesi_yap, yeni_nokta_grid, yeni_nokta_aktif, dogrusallastirma_hatasi
import knapsack

# ─── Toy function for tests: kink at x = 1, minimum value -1 ───
def oyuncak_f(x):
    return max(-2 * x + 1, x - 2)

def oyuncak_subgradient(x):
    if x <= 1:
        return -2
    else:
        return 1


def test_kink_minimum_bulma():
    # The algorithm should find the toy function's minimum (x = 1)
    gecmis = bundle_calistir(oyuncak_f, oyuncak_subgradient, 3, 0.1, 30, yazdir=False)
    assert abs(gecmis[-1] - 1) < 0.001


def test_tek_kesi_subgradient_adimi():
    # With a single cut, the candidate equals a subgradient step (Project 1): center - t * slope
    demet = [kesi_yap(3, oyuncak_f, oyuncak_subgradient)]   # slope 1
    assert yeni_nokta_aktif(demet, 3, 1) == 2


def test_grid_aktif():
    # Referee test: our solver should find the same candidate as the grid search
    demet = [kesi_yap(x, oyuncak_f, oyuncak_subgradient) for x in [3, -1, 0.5]]
    fark = yeni_nokta_grid(demet, 0.5, 1) - yeni_nokta_aktif(demet, 0.5, 1)
    assert abs(fark) < 0.01


def test_knapsack_dual_kagit_degerleri():
    # Values computed by hand on paper
    assert knapsack.f(0) == 20
    assert knapsack.f(1) == 15
    assert knapsack.f(2) == 14
    assert knapsack.subgradient(1.5) == -1
    assert knapsack.subgradient(2.5) == 4


def test_knapsack_dual_minimumu():
    # Bundle should find the dual's minimum at λ = 2; upper bound 14, above the true optimum (10)
    gecmis = bundle_calistir(knapsack.f, knapsack.subgradient, 3, 0.1, 30, yazdir=False)
    lam = gecmis[-1]
    assert abs(lam - 2) < 0.001
    assert abs(knapsack.f(lam) - 14) < 0.001
    assert knapsack.f(lam) >= 10   # weak duality
    
def test_dogrusallastirma_hatasi():
    kesi = kesi_yap(3, knapsack.f, knapsack.subgradient)
    assert dogrusallastirma_hatasi(kesi, 3, knapsack.f) == 0
    assert dogrusallastirma_hatasi(kesi, 2, knapsack.f) == 2