from scipy.optimize import minimize_scalar
def f(x):
    # Köşeli fonksiyon: x=1'de kırılma (kink) var. Minimum noktası x=1, değeri -1.
    return max(-2 * x + 1, x - 2)

def subgradient(x):
    if x <= 1: 
        return -2  # Sol tarafın eğimi
    # doğru denklemi $-2x + 1$'dir. Denklemin katsayısına baktığımızda bu bölgedeki türevin (eğimin) -2 olduğunu görürüz
    else:
        return 1   # Sağ tarafın eğimi

    
def kesi_yap(xi):
    return (xi, f(xi), subgradient(xi)) 

def kesi_degeri(kesi, x):
    xi, fi, gi = kesi
    return fi + gi * (x - xi)

def model(demet, x):
    # Elimizdeki tüm kesilerin x'teki değerlerini hesapla, en büyüğünü seç.
    # Bu bize fonksiyonun alt sınır (yaklaşık) modelini verir.
    return max(kesi_degeri(k, x) for k in demet)  
def yeni_nokta_grid(demet, merkez, t):
    # Hakem: model + cezayı merkez±10 aralığında 0.01 adımla tarar (yavaş ama güvenilir)
    en_iyi_x = merkez
    en_kucuk = model(demet, merkez)
    x = merkez - 10
    while x <= merkez + 10:
        toplam = model(demet, x) + ((x - merkez) ** 2) / (2 * t)
        if toplam < en_kucuk:
            en_kucuk = toplam
            en_iyi_x = x
        x = x + 0.01
    return en_iyi_x 

def yeni_nokta_scipy(demet, merkez, t):
    def hedef(x):
        return model(demet, x) + ((x - merkez) ** 2) / (2 * t)
    #modelin degeri + merkezden uzaklasma cezasi

    sonuc = minimize_scalar(hedef, bounds=(merkez - 10, merkez + 10), method='bounded')
    #x minimumu bulyor. merkez -10 10 araliginda bound belirliyoruz 
    return sonuc.x
#sonuc.x: minimize_scalar bir obje döndürür
#  .x özelliği, fonksiyonu minimum yapan ideal x değeridir.

def bundle_calistir(baslangic_merkez, baslangic_t, max_tur):
    merkez = baslangic_merkez
    t = baslangic_t
    demet = [kesi_yap(merkez)] # Hiç kesi olmadan model olmaz, başlangıç ölçümünü ekliyoruz.

    for i in range(max_tur):
        
        # Demeti, merkezi ve t'yi kullanarak modelin bize önerdiği en iyi aday noktayı (x) bul.
        x = yeni_nokta_scipy(demet, merkez, t)
        
        # ADIM 2: DURMA KURALI (VAAT KONTROLÜ)
        # Mevcut konumumuz ile modelin x'te vaat ettiği değer arasındaki fark çok küçükse dur.
        vaat = f(merkez) - model(demet, x)
        if vaat < 0.000001:
            print(f"\n--> Algoritma durdu! Tur: {i}, Merkez: {merkez}")
            break
            
        
        # Gittiğimiz x noktasında gerçek durumu ölç ve sonucu ne olursa olsun demete ekle.
        demet.append(kesi_yap(x))
        
        #{f}(x) + {1}/ 2t} (x - xc) kare  
        # Aday nokta x, gerçekten şu anki merkezimizden daha mı alçakta?
        if f(x) < f(merkez):  
            # Evet, daha alçakta! Merkezimizi x'e taşıyalım (Serious Step)
            t = t * 2
            merkez = x
            print(f"Tur {i:2d} | x: {x:.4f} | t: {t} | Karar: SERIOUS (Taşındı)")
        else:
            t=t/ 2
            # Hayır. Merkez yerinde kalır ama demet yeni bir kesi öğrendiği için model akıllandı (Null Step)
            print(f"Tur {i:2d} | x: {x:.4f} | t: {t} | Karar: NULL (Sabit kaldı)")
            
    #
    # Döngü bittikten (veya break ile kırıldıktan) sonra ulaşılan en iyi merkezi geri döndür.
    return merkez

if __name__ == "__main__":
    print("Algoritma Başlıyor...\n" + "-"*45)
    sonuc = bundle_calistir(3, 0.1, 30)
    print("-" * 45)
    print("Bulunan minimum noktası:", sonuc)

    demet = [kesi_yap(3)]
    print("grid :", yeni_nokta_grid(demet, 3, 1))
    print("scipy:", yeni_nokta_scipy(demet, 3, 1))