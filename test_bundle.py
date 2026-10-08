from bundle import bundle_calistir

def test_kink_minimum_bulma():
    # Başlangıç: merkez 3, t 0.1, max tur 30
    sonuc = bundle_calistir(3, 0.1, 30)
    
    # Sonucun 1'e (kırılma noktasına) çok yakın olmasını bekliyoruz.
    # Sonucun 1'den farkının (mesafesinin) 0.05'ten küçük olup olmadığını test ediyoruz.
    assert abs(sonuc - 1) < 0.05