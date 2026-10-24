# Mini bundle method (one-dimensional)
# Finds the minimum of any nonsmooth convex function.
# The function is supplied from OUTSIDE: f and subgradient are passed as parameters.
# This file contains no problem, only the algorithm.


# ─────────────────────────────────────────────
# 1) Cuts and the model (the map)
# ─────────────────────────────────────────────

def kesi_yap(xi, f, subgradient):
    # Evaluate at x = xi and pack three numbers: (point, value, slope)
    # The function to evaluate (f, subgradient) is passed in from outside.
    return (xi, f(xi), subgradient(xi))


def kesi_degeri(kesi, x):
    # Build the line from the cut and return its height at x:
    # cut(x) = value + slope * (x - point)
    xi, fi, gi = kesi
    return fi + gi * (x - xi)


def model(demet, x):
    # The maximum height of all cuts at x.
    # An (optimistic) map that always stays below the true f.
    return max(kesi_degeri(k, x) for k in demet)


# ─────────────────────────────────────────────
# 2) Master problem: "where should I go next?"
#    minimize: model(x) + (x - center)^2 / (2t)
#    Note: this part NEVER sees the true f, only the cuts in the bundle.
# ─────────────────────────────────────────────

def yeni_nokta_aktif(demet, merkez, t):
    # Our own solver: list all sensible candidates, pick the lowest-scoring one.

    # Part 0: scoring function (map + leash)
    def hedef(x):
        return model(demet, x) + (x - merkez) ** 2 / (2 * t)

    adaylar = []

    # Part 1: "if only this cut were active" -> center - t * slope
    for kesi in demet:
        xi, fi, gi = kesi
        adaylar.append(merkez - t * gi)

    # Part 2: "if two cuts were active together" -> intersection of the two lines
    for i in range(len(demet)):
        for j in range(i + 1, len(demet)):
            nokta1, deger1, egim1 = demet[i]
            nokta2, deger2, egim2 = demet[j]
            if egim1 == egim2:
                continue   # parallel lines never intersect
            sabit1 = deger1 - egim1 * nokta1   # line: egim1 * x + sabit1
            sabit2 = deger2 - egim2 * nokta2   # line: egim2 * x + sabit2
            adaylar.append((sabit2 - sabit1) / (egim1 - egim2))

    # Part 3: pick the candidate with the lowest score
    en_iyi = adaylar[0]
    for x in adaylar:
        if hedef(x) < hedef(en_iyi):
            en_iyi = x
    return en_iyi


def yeni_nokta_grid(demet, merkez, t):
    # REFEREE: solves the same problem by scanning center ± 10 in steps of 0.01.
    # Slow but fully transparent; used only in tests.
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
# 3) Main algorithm
# ─────────────────────────────────────────────

def bundle_calistir(f, subgradient, baslangic_merkez, baslangic_t, max_tur, yazdir=True):
    # f, subgradient: the function to minimize and its slope (from outside)
    # Returns: the list of centers per iteration (last element = minimum found)
    merkez = baslangic_merkez
    t = baslangic_t
    demet = [kesi_yap(merkez, f, subgradient)]   # first evaluation
    gecmis = [merkez]

    for i in range(max_tur):
        # CANDIDATE: propose the next point using the map and the leash
        x = yeni_nokta_aktif(demet, merkez, t)

        # STOP?: if the map (even though optimistic) promises no improvement, stop
        vaat = f(merkez) - model(demet, x)
        if vaat < 0.000001:
            if yazdir:
                print(f"--> Algoritma durdu! Tur: {i}, Merkez: {merkez}")
            break

        # EVALUATE: go to the candidate, measure true value and slope, add to the bundle
        demet.append(kesi_yap(x, f, subgradient))

        # DECISION: is the candidate really lower?
        if f(x) < f(merkez):
            # Serious step: move, the map was reliable -> lengthen the leash
            merkez = x
            t = t * 2
            karar = "SERIOUS (Taşındı)"
        else:
            # Null step: stay, the map was wrong -> shorten the leash
            t = t / 2
            karar = "NULL (Sabit kaldı)"

        gecmis.append(merkez)
        if yazdir:
            print(f"Tur {i:2d} | x: {x:.4f} | t: {t} | Karar: {karar}")

    return gecmis


def dogrusallastirma_hatasi(kesi, merkez, f):
    # Linearization error: how far the cut lies below the true f at the center (always >= 0)
    # 0 = fresh cut, large = stale cut
    return f(merkez) - kesi_degeri(kesi, merkez)