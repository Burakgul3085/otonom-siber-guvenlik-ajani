# cyber security agent - cicids2017 anomali tespiti

Bu proje, CICIDS2017 veri seti uzerinde **sadece normal trafik ile egitilen** bir `Dense Autoencoder` yaklasimi kullanarak ag trafigindeki saldiri anomalilerini tespit eden ve tespit aninda otonom aksiyon (dummy IP bloklama) alabilen bir siber guvenlik ajani gelistirmek amaciyla hazirlanmistir.

> Bu calismada **transfer learning** veya hazir egitilmis mimariler kesinlikle kullanilmamistir. Tum model katmanlari Keras ile sifirdan kurulmustur.

---

## 1) projenin amaci

Geleneksel imza tabanli IDS yaklasimlari, bilinmeyen ve varyant saldirilar karsisinda yetersiz kalabilir. Bu projede hedef:

- normal davranis kalibini ogrenmek,
- normalden sapmalari yeniden uretim hatasi (MSE) uzerinden olcmek,
- esik ustu anomalileri saldiri olarak siniflandirmak,
- tespit aninda otomatik bir koruma aksiyonu tetiklemek,
- tum sureci modern bir dashboard ile gozlemlenebilir hale getirmektir.

Bu sayede sistem, hem akademik olarak aciklanabilir hem de prototip seviyesinde operasyonel bir guvenlik ajani davranisi sergiler.

---

## 2) kullanilan teknolojiler

- `python 3.10+`
- `pandas`, `numpy` (veri isleme)
- `scikit-learn` (normalizasyon, metrikler, split)
- `tensorflow / keras` (sifirdan autoencoder mimarisi)
- `matplotlib`, `seaborn` (grafikler ve confusion matrix)
- `streamlit` (etkilesimli dashboard arayuzu)

---

## 3) veri seti ve dengeleme stratejisi

### kaynak veri
`data/` klasorunde bulunan CICIDS2017 CSV dosyalari icinden **cuma gunu** trafik dosyalari okunur:

- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`

### on isleme adimlari
`data_preprocessing.py` dosyasi su adimlari uygular:

1. Cuma CSV dosyalarini birlestirme
2. `NaN`, `+Infinity`, `-Infinity` degerlerini temizleme
3. Kategorik sutunlari sayisallastirma
4. Etiketleri binary yapiya donusturme:
   - `BENIGN -> 0 (Normal)`
   - `DIGER TUM ETIKETLER -> 1 (Saldiri)`
5. Veri setini dengeli hale getirme:
   - `100.000 Normal`
   - `100.000 Saldiri`
6. Tum ozelliklerde `MinMaxScaler` ile `[0,1]` araligina olcekleme

### neden dengeleme?
Sinif dengesizligi, modelin sadece cogunluk sinifina ogrenme kaymasina neden olur. 100k-100k dengeleme ile:

- model egitim ve degerlendirme asamalarinda adil bir karar siniri gorur,
- precision/recall dengesi iyilesir,
- threshold optimizasyonu daha anlamli hale gelir.

---

## 4) autoencoder mimarisi ve mantigi

`train_autoencoder.py` dosyasinda model sifirdan olusturulur:

- **encoder:** `Dense(128) -> Dropout -> Dense(64) -> Dropout -> Dense(32 latent)`
- **decoder:** `Dense(64) -> Dropout -> Dense(128) -> Dense(input_dim)`
- kayip fonksiyonu: `MSE`
- optimizer: `Adam`

### neden sadece normal veriyle egitim?
Autoencoder, girdiyi yeniden insa etmeyi ogrenir. Eger sadece normal trafikle egitilirse:

- normal ornekleri dusuk hata ile yeniden uretir,
- saldiri trafikleri normal dagilimdan saptigi icin yuksek MSE uretir,
- bu hata farki threshold tabanli anomali tespitini guclendirir.

Bu strateji one-class / semi-supervised anomaly detection paradigmasina uygundur.

### threshold ve model secimi nasil yapiliyor?
Egitim asamasinda yalnizca tek bir model degil, birden fazla aday model ve farkli seed kombinasyonlari degerlendirilir:

- 3 farkli dense autoencoder adayi egitilir,
- her aday `3 seed` ile (`42`, `77`, `123`) tekrar edilir,
- secim politikasi:
  - once `recall >= 0.80` kosulunu saglayan deneyler filtrelenir,
  - bu kümeden en yuksek `F1` secilir,
  - hicbiri kosulu saglamazsa en yuksek `recall` secilir (esitlikte F1).

Threshold tarafinda validation MSE dagilimi uzerinden hem dengeli (F1) hem recall oncelikli (F2) esik hesaplanir. Uretim/simulasyon tarafinda varsayilan olarak recall oncelikli esik kullanilir. Bu esikler:

- `artifacts/threshold.json` dosyasina kaydedilir,
- dashboard ajaninda karar mekanizmasi olarak kullanilir.

Ayrica validation MSE histogrami:

- `artifacts/validation_mse_distribution.png`

olarak uretilir.

---

## 5) otonom ajan ve dashboard ozellikleri

`agent_dashboard.py` dosyasi Streamlit tabanli bir GUI sunar.

### arayuzde bulunan ana moduller
- canli paket log alani (paket paket simulasyon akisi)
- anlik MSE cizgi grafigi
- durum gostergesi:
  - threshold alti: **guvenli**
  - threshold ustu: **saldiri tespit edildi - ip bloklaniyor**
- simulasyon sonu metrik paneli:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
- karar paneli:
  - threshold carpani (recall/precision dengesi icin)
  - pencere tabanli karar filtresi (opsiyonel)

### otonom tepki mekanizmasi
Saldiri tespitinde `block_ip()` fonksiyonu tetiklenir. Dashboard tarafinda iki mod desteklenir:

- `dry-run` (varsayilan, guvenli): gercek bloklama yapmaz, calistirilacak firewall komutunu loglar.
- gercek komut modu: Windows ortaminda `netsh advfirewall`, Linux ortaminda `iptables` komutunu calistirir.

Bu yaklasim, laboratuvar/sunum ortaminda guvenli test imkani saglarken, ihtiyac halinde gercek firewall otomasyonuna gecisi kolaylastirir.

---

## 6) klasor yapisi

```text
DerinProje/
├─ data/                              # ham veri (git disi)
├─ artifacts/                         # uretilen ciktilar
├─ .gitignore
├─ requirements.txt
├─ data_preprocessing.py
├─ train_autoencoder.py
├─ agent_dashboard.py
├─ evaluate.py
└─ README.md
```

---

## 7) kurulum ve calistirma

### 1) bagimliliklar
```bash
pip install -r requirements.txt
```

### 2) veri on isleme
```bash
python data_preprocessing.py
```

### 3) model egitimi ve threshold hesaplama
```bash
python train_autoencoder.py
```

Bu adim sonunda:
- `model.h5`
- `artifacts/threshold.json`
- `artifacts/test_set.csv`
- `artifacts/validation_mse_distribution.png`

olusturulur.

### 4) dashboard ajanini baslatma
```bash
streamlit run agent_dashboard.py
```

### 5) otomatik threshold/pencere tuning
```bash
python evaluate.py --target-recall 0.80
```

Bu komut:
- farkli threshold carpani + pencere kombinasyonlarini tarar,
- en iyi ayari secer,
- `artifacts/evaluation_results.csv` ve `artifacts/best_config.json` dosyalarini uretir.

---

## 8) yeniden uretilebilirlik ve iyi muhendislik pratikleri

- rastgelelik kontrolu (`random_state`) sabit tutulmustur.
- veri sizintisini azaltmak icin egitim yalnizca normal sinifta yapilir.
- threshold secimi validation tabanli yapilir.
- model secimi tek kosu yerine coklu aday + coklu seed ile yapilir.
- kod OOP odakli, moduler ve acik sorumluluklara bolunmustur.
- model agirliklari ve ham veri dosyalari `.gitignore` ile korunmustur.

---

## 9) nihai deney sonucu (proje final konfigurasyonu)

Asagidaki konfigurasyon ile 1000 paketlik test simulasyonunda elde edilen nihai sonuc:

- secili model: `recall_dense`
- secili seed: `123`
- threshold stratejisi: `recall_priority`
- dashboard ayarlari:
  - threshold carpani: `1.00`
  - karar penceresi: `1`
  - pencerede minimum saldiri oyu: `1`
  - maksimum paket: `1000`
  - gecikme: `0.01 sn`

### performans metrikleri
- Accuracy: `0.8470`
- Precision: `0.8535`
- Recall: `0.8451`
- F1-Score: `0.8493`

### confusion matrix
- TN: `416`
- FP: `74`
- FN: `79`
- TP: `431`

Bu sonuc, saldiri yakalama kabiliyeti (recall) ile yanlis alarm kontrolu (precision) arasinda guclu bir denge saglandigini gostermektedir.

---

## 10) gelistirme icin sonraki adimlar

- gercek zamanli paket yakalama (pcap/socket) entegrasyonu
- dinamik threshold (zaman pencereli adaptif yapi)
- SHAP/feature-attribution ile aciklanabilirlik katmani
- dummy `block_ip()` yerine gercek firewall otomasyonu
- CI/CD ve test altyapisi (unit + smoke test)

---

## 11) lisans ve akademik not

Bu repo, bitirme/donem projesi kapsaminda egitsel ve arastirma odakli bir prototip olarak tasarlanmistir. Uretim ortamlarinda dogrudan kullanmadan once guvenlik sertlestirmesi, olceklenebilirlik testleri ve mavi-takim/kirmizi-takim dogrulamalari yapilmalidir.
