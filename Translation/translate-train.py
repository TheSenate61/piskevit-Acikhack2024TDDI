from datasets import load_dataset
from deep_translator import GoogleTranslator
import pandas as pd
from tqdm import tqdm

# Veri kümesini yükle
# PKU-Alignment/PKU-SafeRLHF-30K veri kümesini yükler. 
# Bu veri kümesi, model güvenliği ile ilgili yanıtlar içeren bir veri kümesidir.
dataset = load_dataset('PKU-Alignment/PKU-SafeRLHF-30K')  # Gerçek veri kümesinin adını buraya yazın

# Çeviri işlemi için Google Translator'ı başlat
# GoogleTranslator sınıfını kullanarak, İngilizce'den Türkçe'ye çeviri yapmak için bir çevirmen oluşturulur.
translator = GoogleTranslator(source='en', target='tr')

# Metni çeviren fonksiyon
# Bu fonksiyon, verilen İngilizce metni Türkçe'ye çevirir.
def translate_text(text):
    return translator.translate(text)

# Satırları çeviren fonksiyon
# Bu fonksiyon, belirli sütunlardaki metinleri çevirir. 
# `rows` parametresi, çevrilecek satırları içerir. 
# `columns` parametresi ise hangi sütunların çevrileceğini belirtir.
def translate_rows(rows, columns):
    translated_rows = []  # Çevrilmiş satırları tutmak için bir liste oluşturulur
    for row in tqdm(rows, desc="Translating rows"):  # Satırlar üzerinde iterasyon yapılır, ilerleme durumu `tqdm` ile gösterilir
        translated_row = row.copy()  # Orijinal satırı değiştirmemek için bir kopyası oluşturulur
        for column in columns:  # Belirtilen sütunlar üzerinde iterasyon yapılır
            if column in translated_row and translated_row[column]:  # Sütun, satırda mevcutsa ve boş değilse
                translated_row[column] = translate_text(translated_row[column])  # Sütundaki metin çevrilir
        translated_rows.append(translated_row)  # Çevrilmiş satır listeye eklenir
    return translated_rows  # Çevrilmiş satırlar geri döndürülür

# Filtrelenmiş satırları çeviren ve kaydeden fonksiyon
# Bu fonksiyon, belirli bir başlangıç satırından itibaren `n` adet satırı seçer,
# güvenli olmayan yanıtları filtreler ve çevrilen sonuçları CSV dosyalarına kaydeder.
def translate_and_save_filtered_rows(dataset, start_row=300, n=20000, save_interval=100):
    # Çevrilecek sütunları tanımla
    columns_to_translate = ['prompt', 'response_0', 'response_1']
    
    # Eğitim ve test veri setlerini çevir
    for split in ['train', 'test']:
        if split in dataset:  # Veri setinde ilgili bölüm varsa
            split_data = dataset[split]  # İlgili veri bölümü seçilir
            
            # Başlangıç satırından başlayarak `n` satır seçilir
            data_list = split_data.select(range(start_row, min(start_row + n, len(split_data)))).to_pandas()

            # Her iki güvenlik sütununun da `False` olduğu satırlar filtrelenir
            filtered_data = data_list[~((data_list['is_response_0_safe'] == False) & (data_list['is_response_1_safe'] == False))]

            # Filtrelenmiş satırlar, sözlükler listesine dönüştürülür
            filtered_rows = filtered_data.to_dict(orient='records')

            # Çevrilmiş satırları tutmak için bir liste başlatılır
            all_translated_rows = []

            # Filtrelenmiş satırları belirli aralıklarla çevirip kaydeder
            for i in tqdm(range(0, len(filtered_rows), save_interval), desc="Processing in chunks"):
                chunk = filtered_rows[i:i + save_interval]  # Çevrilecek satırların bir parçası seçilir
                translated_chunk = translate_rows(chunk, columns_to_translate)  # Parça çevrilir
                all_translated_rows.extend(translated_chunk)  # Çevrilen parça listeye eklenir
                
                # Çevrilmiş satırlar DataFrame'e dönüştürülür
                translated_df = pd.DataFrame(translated_chunk)
                
                # Çevrilen DataFrame bir CSV dosyasına kaydedilir
                translated_df.to_csv(f"translated_{split}_filtered_rows_part_{i//save_interval + 82}.csv", index=False)
            
            # İsteğe bağlı olarak, tam çevrilmiş DataFrame'i kaydet
            all_translated_df = pd.DataFrame(all_translated_rows)
            all_translated_df.to_csv(f"translated_{split}_filtered_rows_complete.csv", index=False)

# Filtrelenmiş satırları çevir ve CSV dosyası olarak kaydet
translate_and_save_filtered_rows(dataset, start_row=8000, n=20000, save_interval=100)
