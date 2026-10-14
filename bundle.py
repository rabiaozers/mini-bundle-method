# Mini bundle method (tek boyutlu)
# Köşeli bir fonksiyonun minimumunu, kesileri biriktirip
# "model + tasma" problemini kendi aktif-küme çözücümüzle çözerek bulur.


# ─────────────────────────────────────────────
# 1) Problem: minimumunu aradığımız fonksiyon
# ─────────────────────────────────────────────

def f(x):
    # Köşeli fonksiyon: x = 1'de kırılma var. Minimum noktası x = 1, değeri -1.
    return max(-2 * x + 1, x - 2)


def subgradient(x):
    # Köşenin hangi tarafındayım? O taraftaki doğrunun eğimini döndür.
    # Tam köşede (x = 1) alt diferansiyel [-2, 1]; içinden -2'yi seçiyoruz.
    if x <= 1:
        return -2   # sol taraf: -2x + 1
    else:
        return 1    # sağ taraf: x - 2


# ─────────────────────────────────────────────
# 2) Kesiler ve model (harita)
# ─────────────────────────────────────────────

def kesi_yap(xi):
    # x = xi noktasında ölçüm yap, üç sayıyı paketle: (nokta, değer, eğim)
    return (xi, f(xi), subgradient(xi))


def kesi_degeri(kesi, x):
    # Paketten doğruyu üret ve x'teki yüksekliğini döndür:
    # kesi(x) = değer + eğim * (x - nokta)
    xi, fi, gi = kesi
    return fi + gi * (x - xi)


def model(demet, x):
    # Bütün kesilerin x'teki yüksekliklerinin en büyüğü.
    # Gerçek f'nin hep altında kalan (iyimser) bir harita.
    return max(kesi_degeri(k, x) for k in demet)


# ─────────────────────────────────────────────
# 3) Master problem: "bir sonraki nereye gideyim?"
#    küçült: model(x) + (x - merkez)^2 / (2t)
# ─────────────────────────────────────────────

def yeni_nokta_aktif(demet, merkez, t):
    # Kendi çözücümüz: bütün mantıklı adayları listele, en düşük puanlıyı seç.

    # Parça 0: puan makinesi (harita + tasma)
    def hedef(x):
        return model(demet, x) + (x - merkez) ** 2 / (2 * t)

    adaylar = []

    # Parça 1: "sadece bu kesi aktif olsaydı" -> merkez - t * eğim
    for kesi in demet:
        xi, fi, gi = kesi
        adaylar.append(merkez - t * gi)

    # Parça 2: "iki kesi birden aktif olsaydı" -> iki doğrunun kesişimi
    for i in range(len(demet)):
        for j in range(i + 1, len(demet)):
            nokta1, deger1, egim1 = demet[i]
            nokta2, deger2, egim2 = demet[j]
            if egim1 == egim2:
                continue   # paralel doğrular, kesişmezler
            sabit1 = deger1 - egim1 * nokta1   # doğru: egim1 * x + sabit1
            sabit2 = deger2 - egim2 * nokta2   # doğru: egim2 * x + sabit2
            adaylar.append((sabit2 - sabit1) / (egim1 - egim2))

    # Parça 3: puanı en küçük adayı seç
    en_iyi = adaylar[0]
    for x in adaylar:
        if hedef(x) < hedef(en_iyi):
            en_iyi = x
    return en_iyi


def yeni_nokta_grid(demet, merkez, t):
    # HAKEM: aynı problemi merkez ± 10 aralığında 0.01 adımla tarayarak çözer.
    # Yavaş ama içi tamamen bilinen bir yöntem; sadece testlerde kullanılır.
    en_iyi_x = merkez
    en_kucuk = model(demet, merkez)
    x = merkez - 10
    while x <= merkez + 10:
        toplam = model(demet, x) + (x - merkez) ** 2 / (2 * t)
        if toplam < en_kucuk:
            en_kucuk = toplam
            en_iyi_x = x
        x = x + 0.01
    return en_iyi_x


def bundle_calistir(baslangic_merkez, baslangic_t, max_tur):
    merkez = baslangic_merkez
    t = baslangic_t
    demet = [kesi_yap(merkez)]   # hiç kesi olmadan model olmaz: ilk ölçüm
    gecmis = [merkez]
    for i in range(max_tur):
        # ADAY: haritaya ve tasmaya bakarak bir sonraki noktayı öner
        x = yeni_nokta_aktif(demet, merkez, t)

        # DUR MU?: harita (iyimser olduğu halde) iyileşme vaat etmiyorsa dur
        vaat = f(merkez) - model(demet, x)
        if vaat < 0.000001:
            print(f"\n--> Algoritma durdu! Tur: {i}, Merkez: {merkez}")
            break

        # ÖLÇ: adaya git, gerçek değeri ve eğimi ölç, deftere ekle
        demet.append(kesi_yap(x))

        # KARAR: aday gerçekten daha alçak mı?
        if f(x) < f(merkez):
            # Serious step: taşın, harita güvenilir çıktı -> tasmayı uzat
            merkez = x
            t = t * 2
            print(f"Tur {i:2d} | x: {x:.4f} | t: {t} | Karar: SERIOUS (Taşındı)")
            gecmis.append(merkez)
        else:
            # Null step: yerinde kal, harita yanıldı -> tasmayı kısalt
            t = t / 2
            print(f"Tur {i:2d} | x: {x:.4f} | t: {t} | Karar: NULL (Sabit kaldı)")
            gecmis.append(merkez)

    return gecmis


if __name__ == "__main__":
    print("Algoritma Başlıyor...\n" + "-" * 45)
    sonuc = bundle_calistir(3, 0.1, 30)
    print("-" * 45)
    print("Bulunan minimum noktası:", sonuc[-1])