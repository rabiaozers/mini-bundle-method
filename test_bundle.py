from bundle import bundle_calistir, kesi_yap, yeni_nokta_grid, yeni_nokta_aktif


def test_kink_minimum_bulma():
    # Algoritma köşeli fonksiyonun minimumunu (x = 1) bulmalı
    sonuc = bundle_calistir(3, 0.1, 30)
    assert abs(sonuc[-1] - 1) < 0.001


def test_tek_kesi_subgradient_adimi():
    # Tek kesiyle aday, Proje 1'deki adımla aynı olmalı: merkez - t * eğim
    demet = [kesi_yap(3)]          # eğim 1
    assert yeni_nokta_aktif(demet, 3, 1) == 2


def test_grid_aktif():
    # Hakem testi: kendi çözücümüz, grid ile aynı aday noktayı bulmalı
    demet = [kesi_yap(3), kesi_yap(-1), kesi_yap(0.5)]
    merkez = 0.5
    t = 1
    fark = yeni_nokta_grid(demet, merkez, t) - yeni_nokta_aktif(demet, merkez, t)
    assert abs(fark) < 0.01