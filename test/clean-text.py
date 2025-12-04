import unittest
import pandas as pd
# from your_module import clean_text # Gerçek uygulamada import edilmesi gereken kısım
# clean_text fonksiyonunu manuel olarak buraya eklediğinizi varsayalım:
def clean_text(text: str) -> str:
    import re
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]*>", " ", text) # HTML etiketleri
    text = re.sub(r"http\S+|www\.\S+", " ", text) # URL'ler
    text = re.sub(r"\([^)]*\)", " ", text) # Parantez içi
    text = re.sub(r'[^a-z\s]', ' ', text) # Rakamlar, noktalamalar, özel karakterler
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS] # Stop words
    text = " ".join(tokens)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class TestDataCleaning(unittest.TestCase):
    def test_clean_text_full_pipeline(self):
        """Çeşitli temizleme adımlarını tek bir metinde test eder."""
        input_text = "This is a **Test** sentence with <a href='link'>HTML</a> tags, a URL like http://example.com, and some numbers: 123 (and parenthesized content). We should remove stop words."
        expected_output = "test sentence html tags url like numbers remove stop words"        
        actual_output = clean_text(input_text)
        
        # Beklenen çıktı ile karşılaştır
        self.assertEqual(actual_output, expected_output)

    def test_clean_text_non_string_input(self):
        """Dize olmayan girişlerin boş dize döndürüp döndürmediğini test eder."""
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(12345), "")



# from your_module import load_and_clean # Gerçek uygulamada import edilmesi gereken kısım

# load_and_clean fonksiyonunu simüle etmek için:
def mock_load_and_clean(path, label):
    # ... (data sözlüğü oluşturma kısmı aynı kalmalı)
    if path == "mock_human.csv":
        data = {'text': ['sample text 1', 'sample text 2'], 
                'extra_col': [1, 2]} 
    else:
        data = {'text': ['sample text 3', 'sample text 4'], 
                'extra_col': [3, 4]}

    # Önce tüm kolonlarla DataFrame oluşturulur:
    df = pd.DataFrame(data)
    
    # Ardından, sadece gerekli olan 'text' kolonunu seçerek sızıntı önleme simüle edilir:
    # Bu, usecols kullanma amacınızı yerine getirir.
    df = df[['text']].copy() 
    
    # text kolonunu temizle
    df["text"] = df["text"].astype(str).apply(clean_text) 
    df["label"] = label
    # ... (Geri kalan kod aynı kalmalı)
    df = df[df["text"].str.len() > 5] 
    return df
class TestDataLoading(unittest.TestCase):
    def test_column_filter_and_labeling(self):
        """Fonksiyonun sadece 'text' kolonunu yüklediğini ve doğru etiketlediğini test eder."""
        mock_path = "mock_human.csv"
        label = "human"
        
        # Mock fonksiyonu çağır
        result_df = mock_load_and_clean(mock_path, label)
        
        # 1. Kolon kontrolü (Sadece 'text' ve 'label' olmalı)
        self.assertListEqual(list(result_df.columns), ['text', 'label'])
        
        # 2. Etiket kontrolü (Tüm satırların doğru etikete sahip olması)
        self.assertTrue((result_df['label'] == label).all())
        
        # 3. Satır sayısı kontrolü (Yükleme/filtreleme sonrası)
        self.assertGreater(len(result_df), 0)

from sklearn.utils import shuffle
# from your_module import main # Gerçek uygulamada import edilmesi gereken kısım

class TestMainLogic(unittest.TestCase):
    def test_shuffling_is_consistent(self):
        """Veri karıştırma işleminin (shuffle) random_state ile tutarlı olup olmadığını test eder."""
        data = {'text': [f'text_{i}' for i in range(10)], 'label': [0] * 5 + [1] * 5}
        df = pd.DataFrame(data)
        
        # Aynı random_state ile ilk karıştırma
        shuffled_df_1 = shuffle(df, random_state=42).reset_index(drop=True)
        
        # Aynı random_state ile ikinci karıştırma
        shuffled_df_2 = shuffle(df, random_state=42).reset_index(drop=True)
        
        # 1. Karıştırma işleminin yapılıp yapılmadığını kontrol et (ilk satır farklı olmalı)
        self.assertFalse(df.equals(shuffled_df_1)) 

        # 2. Tutarlılık kontrolü (Aynı random_state ile iki çıktı aynı olmalı)
        self.assertTrue(shuffled_df_1.equals(shuffled_df_2))




# from your_module import main # Gerçek uygulamada import edilmesi gereken kısım

class TestMainLogic(unittest.TestCase):
    def test_shuffling_is_consistent(self):
        """Veri karıştırma işleminin (shuffle) random_state ile tutarlı olup olmadığını test eder."""
        data = {'text': [f'text_{i}' for i in range(10)], 'label': [0] * 5 + [1] * 5}
        df = pd.DataFrame(data)
        
        # Aynı random_state ile ilk karıştırma
        shuffled_df_1 = shuffle(df, random_state=42).reset_index(drop=True)
        
        # Aynı random_state ile ikinci karıştırma
        shuffled_df_2 = shuffle(df, random_state=42).reset_index(drop=True)
        
        # 1. Karıştırma işleminin yapılıp yapılmadığını kontrol et (ilk satır farklı olmalı)
        self.assertFalse(df.equals(shuffled_df_1)) 

        # 2. Tutarlılık kontrolü (Aynı random_state ile iki çıktı aynı olmalı)
        self.assertTrue(shuffled_df_1.equals(shuffled_df_2))