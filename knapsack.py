from bundle import bundle_calistir
 
degerler   = [6, 10, 4]
agirliklar = [2, 5, 4]
kapasite   = 6
 
 
def f(x):
    # Dual fonksiyon L(λ): her eşyanın net kazancı (pozitifse) + kira hakkı
    toplam = 0
    for i in range(len(degerler)):
        deger = degerler[i]
        agirlik = agirliklar[i]
        toplam = toplam + max(0, deger - agirlik * x)
    toplam = toplam + kapasite * x
    return toplam
 
 
def subgradient(x):
    # Eğim = kapasite - (net kazancı pozitif olan eşyaların toplam ağırlığı)
    # Negatif: çanta taşıyor, fiyat artmalı. Pozitif: çanta boş kalıyor, fiyat düşmeli.
    kullanilan = 0
    for i in range(len(degerler)):
        deger = degerler[i]
        agirlik = agirliklar[i]
        if deger - agirlik * x > 0:
            kullanilan = kullanilan + agirlik
    return kapasite - kullanilan
 
 
if __name__ == "__main__":
    print("Knapsack dual'ı, bundle method ile")
    print("-" * 45)
    gecmis = bundle_calistir(f, subgradient, 3, 0.1, 30)
    lam = gecmis[-1]
    print("-" * 45)
    print(f"En iyi fiyat λ = {lam}")
    print(f"Üst sınır L(λ) = {f(lam)}   (gerçek en iyi değer: 10)")
 