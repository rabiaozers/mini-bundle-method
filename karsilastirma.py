from bundle import f, subgradient, bundle_calistir
import matplotlib.pyplot as plt
def f(x):
    return max(-2 * x + 1, x - 2)


def subgradient(x):
    if x > 1:       # köşenin sağındaysam
        return 1    # sağdaki doğrunun eğimi
    elif x < 1:     # köşenin solundaysam
        return -2   # soldaki doğrunun eğimi
    else:           # tam köşedeysem
        return 0    # alt diferansiyelden [-2, 1] bir sayı seç


# --- adım kuralları ---

def step_constant(x):
    return 0.2


def step_diminishing(k):
    return 0.2 / (k + 1)


def step_polyak(x, g, fstar):
    if g == 0:      # sıfıra bölmeyi engelle (köşedeysek dur)
        return 0
    else:
        return (f(x) - fstar) / (g ** 2)


# --- algoritma ---

def calistir(kural, x0, adim_sayisi):
    x = x0
    gecmis = [x]
    for k in range(adim_sayisi):
        g = subgradient(x)
        if kural == "sabit":
            alpha = step_constant(x)
        elif kural == "azalan":
            alpha = step_diminishing(k)
        elif kural == "polyak":
            alpha = step_polyak(x, g, -1)  # fstar = -1
        x = x - alpha * g
        gecmis.append(x)
    return gecmis


# --- testler ---

assert f(2) == 0
assert f(0) == 1
assert f(1) == -1
assert subgradient(0.5) == -2, "köşenin solunda eğim -2 olmalı"
assert subgradient(2) == 1, "köşenin sağında eğim 1 olmalı"

x0 = 3
adim = 30

plt.plot(calistir("sabit",  x0, adim), label="subgradient: sabit")
plt.plot(calistir("azalan", x0, adim), label="subgradient: azalan")
plt.plot(calistir("polyak", x0, adim), label="subgradient: Polyak")
plt.plot(bundle_calistir(x0, 0.1, adim), label="bundle", linewidth=3)

plt.axhline(1, color="gray", linestyle="--")
plt.xlabel("tur")
plt.ylabel("x")
plt.legend()
plt.savefig("karsilastirma.png")
plt.show()