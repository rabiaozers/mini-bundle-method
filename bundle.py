# Mini bundle method (tek boyutlu)
# Herhangi bir köşeli, dışbükey fonksiyonun minimumunu bulur.
# Fonksiyonun kendisi DIŞARIDAN verilir: f ve subgradient parametre olarak gelir.
# Bu dosyada hiçbir problem yok, sadece algoritma var.


# ─────────────────────────────────────────────
# 1) Kesiler ve model (harita)
# ─────────────────────────────────────────────

def kesi_yap(xi, f, subgradient):
    # x = xi noktasında ölçüm yap, üç sayıyı paketle: (nokta, değer, eğim)
    # Hangi fonksiyonu ölçeceğini (f, subgradient) dışarıdan alıyor.
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
# 2) Master problem: "bir sonraki nereye gideyim?"
#    küçült: model(x) + (x - merkez)^2 / (2t)
#    Not: bu kısım gerçek f'yi HİÇ görmüyor, sadece demetteki kesileri görüyor.
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


# ─────────────────────────────────────────────
# 3) Ana algoritma
# ─────────────────────────────────────────────

def bundle_calistir(f, subgradient, baslangic_merkez, baslangic_t, max_tur, yazdir=True):
    # f, subgradient: minimumu aranan fonksiyon ve onun eğimi (dışarıdan)
    # Döndürdüğü: her turdaki merkezlerin listesi (son eleman = bulunan minimum)
    merkez = baslangic_merkez
    t = baslangic_t
    demet = [kesi_yap(merkez, f, subgradient)]   # ilk ölçüm
    gecmis = [merkez]

    for i in range(max_tur):
        # ADAY: haritaya ve tasmaya bakarak bir sonraki noktayı öner
        x = yeni_nokta_aktif(demet, merkez, t)

        # DUR MU?: harita (iyimser olduğu halde) iyileşme vaat etmiyorsa dur
        vaat = f(merkez) - model(demet, x)
        if vaat < 0.000001:
            if yazdir:
                print(f"--> Algoritma durdu! Tur: {i}, Merkez: {merkez}")
            break

        # ÖLÇ: adaya git, gerçek değeri ve eğimi ölç, deftere ekle
        demet.append(kesi_yap(x, f, subgradient))

        # KARAR: aday gerçekten daha alçak mı?
        if f(x) < f(merkez):
            # Serious step: taşın, harita güvenilir çıktı -> tasmayı uzat
            merkez = x
            t = t * 2
            karar = "SERIOUS (Taşındı)"
        else:
            # Null step: yerinde kal, harita yanıldı -> tasmayı kısalt
            t = t / 2
            karar = "NULL (Sabit kaldı)"

        gecmis.append(merkez)
        if yazdir:
            print(f"Tur {i:2d} | x: {x:.4f} | t: {t} | Karar: {karar}")

    return gecmis