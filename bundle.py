def f(x):
    return x ** 2

def subgradient(x):
    return 2 * x   #fonksiyon turevlenebilir oldugu icin graident aslinda bu smooth fonksiyon

def kesi_yap(xi):
    return (xi, f(xi), subgradient(xi)) 
#o noktaya git xi=3 tum noktalari al f3 9(fonk degeri) gi=6(egim)  xi=3(bilgiyi aldigimiz nokta)

def kesi_degeri(kesi,x):
    xi, fi, gi = kesi
    return fi + gi * (x - xi)
#kesiyi olusturuyoruz l(x) = f(xi) + gi(x - xi)  kesiyi olusturuyoruz 6x−9

def model(demet,x):
    return max(kesi_degeri(k, x) for k in demet)  
#kesilerden en buyugunu sec 

def yeni_nokta(demet, merkez, t):
    en_iyi_x = merkez
    en_kucuk = model(demet, merkez) 
    #Merkezde (merkez − merkez)² = 0, yani ceza sıfır
    #baslangicta x=3 aldik en iyi =3

    x = merkez - 10 
    #aslında merkezin 10 solundan başlayıp 10 sağına kadar, 0.01'er adımla
    while x <= merkez + 10:
        toplam =  model(demet, x) + (x - merkez) ** 2 / (2 * t)
        #model + proximal ceza 
        #model(demet, x) -- m(x) fonksiyon burada en azindan yaklasik olarak ne seviyede
        # (x-merkez) ** 2 / (2 * t) -- proximal ceza fonksiyonu
        #2t ye bolmek napiyor merkezew daha bagli kalmasini sagliyor 2 ile
        if toplam < en_kucuk:
            en_kucuk = toplam
            en_iyi_x = x
        x = x + 0.01

    return en_iyi_x
    

merkez = 10
t = 0.1
demet = [kesi_yap(merkez)] #merkezde ölçtüğün kesi. Hiç kesi olmadan model olmaz.

for i in range(10):
    x = yeni_nokta(demet, merkez, t)
    #aday noktayi bul
    vaat= f(merkez) - model(demet,x)
    if vaat < 0.000001:
        print("durdu, tur:", i, "merkez:", merkez)
        break
    demet.append(kesi_yap(x))
    #sonuc ne olursa olsun ekle 
    if f(x) < f(merkez):  #aday suan durdugun yerden daha alcak mi alcaksa merkezi aday yani x yap 
        merkez = x
        #t = t * 2 
        print(i, x, t, "serious")
    else:
        #t = t / 2
        print(i, x, t , "null")

    #Tur 0: döngü yeni_nokta'ya 1 kesi + merkez 3 verdi → öneri −3 → kesi eklendi, merkez kaldı.
#Tur 1: döngü yeni_nokta'ya 2 kesi + merkez 3 verdi → bu sefer öneri 0 → kesi eklendi, merkez 0'a taşındı.