# On the knapsack dual: Project 1's subgradient methods vs the bundle method
import matplotlib.pyplot as plt
from bundle import bundle_calistir
from knapsack import f, subgradient

F_YILDIZ = 14   # minimum value of the dual (Polyak needs to know it)


# ─── From Project 1: step rules and subgradient loop ───
def step_constant(x):
    return 0.2

def step_diminishing(k):
    return 0.2 / (k + 1)

def step_polyak(x, g, fstar):
    if g == 0:
        return 0
    return (f(x) - fstar) / (g ** 2)

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
            alpha = step_polyak(x, g, F_YILDIZ)
        x = x - alpha * g
        gecmis.append(x)
    return gecmis


# ─── Comparison ───
x0 = 3
adim = 30

plt.plot(calistir("sabit",  x0, adim), label="subgradient: sabit")
plt.plot(calistir("azalan", x0, adim), label="subgradient: azalan")
plt.plot(calistir("polyak", x0, adim), label="subgradient: Polyak (f* biliniyor)")
plt.plot(bundle_calistir(f, subgradient, x0, 0.1, adim, yazdir=False),
         label="bundle", linewidth=3)

plt.axhline(2, color="gray", linestyle="--")
plt.xlabel("tur")
plt.ylabel("λ (kilo başına fiyat)")
plt.title("Knapsack dual'ı: λ'nın turlara göre değişimi")
plt.legend()
plt.savefig("karsilastirma.png")
plt.show()