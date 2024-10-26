# Kişilik Analizi ile Makine Öğrenimi Modellerinin Performans Değerlendirmesi

Bu proje, kişilik analizi veri setini kullanarak farklı makine öğrenimi modellerinin performansını değerlendirmeyi amaçlamaktadır. Kullanılan yöntemler arasında Lojistik Regresyon, K-En Yakın Komşu (KNN), Karar Ağaçları, Rastgele Orman ve Destek Vektör Makineleri (SVM) bulunmaktadır.

## Proje Aşamaları

1. **Veri Setinin Yüklenmesi**: `data.csv` dosyasından veri seti okunur.
2. **Veri Keşfi**: Veri setinin ilk 5 satırı, istatistiksel bilgileri ve eksik değer kontrolleri yapılır.
3. **Öznitelik Dönüşümü**:
   - Kategorik sütunlar, sayısal değerlere dönüştürülür.
   - Veri normalizasyonu yapılır.
   - Yaş grupları oluşturulur.
4. **Model Eğitimi**: Eğitim ve test setlerine ayrılan veriler ile farklı modeller eğitilir.
5. **Sonuçların Değerlendirilmesi**: Her model için doğruluk, F1 skoru, hassasiyet ve geri çağırma gibi metriklerle sonuçlar değerlendirilir. Ayrıca, her model için karışıklık matrisleri görselleştirilir.

## Elde Edilen Sonuçlar
Model performansları, Rastgele Orman modelinin en yüksek doğruluk oranına sahip olduğunu göstermektedir.

## Kullanım
Projenin çalıştırılması için gerekli kütüphanelerin yüklenmesi gerekmektedir. Aşağıdaki komutları kullanarak gerekli kütüphaneleri yükleyebilirsiniz:

```bash
pip install -r requirements.txt
