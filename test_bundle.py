from bundle import bundle_calistir, kesi_yap, yeni_nokta_grid, yeni_nokta_scipy,yeni_nokta_aktif
def test_kink_minimum_bulma():
    # Başlangıç: merkez 3, t 0.1, max tur 30
    sonuc = bundle_calistir(3, 0.1, 30)
    # Sonucun köşeye (x = 1) çok yakın olmasını bekliyoruz
    assert abs(sonuc - 1) < 0.001

def test_grid_scipy():
    # Hakem testi: iki farklı yöntem aynı aday noktayı bulmalı
    demet = [kesi_yap(3), kesi_yap(-1), kesi_yap(0.5)]
    merkez = 0.5
    t = 1
    fark = yeni_nokta_grid(demet, merkez, t) - yeni_nokta_scipy(demet, merkez, t)
    assert abs(fark) < 0.01

def test_grid_aktif():
    # Hakem testi: kendi çözücümüz, grid ile aynı aday noktayı bulmalı
    demet = [kesi_yap(3), kesi_yap(-1), kesi_yap(0.5)]
    merkez = 0.5
    t = 1
    fark = yeni_nokta_grid(demet, merkez, t) - yeni_nokta_aktif(demet, merkez, t)
    assert abs(fark) < 0.01