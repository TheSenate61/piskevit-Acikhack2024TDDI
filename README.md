# PiskevitSerbestAc-khack2024TDD-


Bu proje, Türkçe dilinde çalışan büyük dil modellerinin güvenliğini artırmak amacıyla geliştirilmiştir. Büyük dil modellerinin zararlı veya istenmeyen çıktılar üretme riski bulunduğundan, özellikle Türkçe dil modelleri için güvenli veri setleri ve benchmark framework'leri oluşturulması kritik öneme sahiptir. 

## Proje Hedefleri

- Türkçe dilinde güvenli dil modelleri geliştirmek.
- Alignment benchmark framework'ü oluşturarak modellerin güvenliğini ölçmek.
- Elle oluşturulmuş ve çevrilmiş veri setleri ile model eğitimi gerçekleştirmek.
- Türkçe diline özgü güvenli veri setleri oluşturarak modeli daha güvenli hale getirmek.

## Proje İş Akışı

1. **Faz I**: Model eğitimi için elle veri hazırlama çalışmaları ve test edilecek modellerin belirlenmesi.
2. **Faz II**: İngilizceden Türkçe'ye çevrilecek veri setlerinin hazırlanması ve çeviri işlemlerinin başlatılması.
3. **Faz III**: Çevrilen verilerle modeller eğitilip veri kalitesinin test edilmesi.
4. **Faz IV**: Benchmark veri seti ve framework'ünün oluşturulması ve modellerin eğitimi.
5. **Faz V**: Benchmarkların gerçekleştirilmesi ve genel instruct girdilerine tepkilerin ölçülmesi.

## Veri Seti

Bu projede kullanılan veri setleri, İngilizce model çıktılarını Türkçeye çevirmek ve elle yazılmış zararlı içeriklerden oluşmaktadır. Veri setimiz, özellikle cinsiyetçilik, ırkçılık ve homofobi gibi zararlı içeriklere karşı güvenli yanıtlar üreten veriler içermektedir.

## Kullanılan Modeller

Proje kapsamında kullanılan modeller:

- **Koç-Digital**
- **YTÜ-Cosmos-Llama**
- **CAL-Llama**

Bu modeller, oluşturulan güvenli veri setleriyle eğitilmiştir.

## Benchmark

Eğitilen modelleri test etmek için üç yöntem kullanılmıştır:

1. İnsan Değerlendirmesi
2. BERT Sınıflandırma Modeli
3. Rouge Skoru

Benchmark sonuçlarına göre modellerin güvenli ve güvensiz yanıt verme oranları aşağıdaki gibidir:

| Model          | UnSafe (Güvensiz) | safe (Güvenli) |
|----------------|----------------|-------------------|
| Koç-Digital    | %48            | %96              |
| YTÜ-Cosmos-Llama | %35           | %94              |
| CAL-Llama      | %8         | %93                |

## Sonuçlar ve Bulgular

- Alignment sürecinden geçirilmemiş Türkçe modeller ileride daha büyük riskler oluşturabilir.
- Alignment verileri modelin genel instruct girdilerine karşı da güvenli yanıtlar vermesini sağlar.
- Güvenli yapay zeka, daha güvenli bir dünya için kritik öneme sahiptir.

## Proje Yol Haritası

- Benchmark framework'ünün Huggingface'de leaderboard olarak sunulması.
- Çalışmanın akademik bir makale olarak yayımlanması.
- Sentetik veri üretimi ile veri setinin derinleştirilmesi ve farklı dillerde denenmesi.
