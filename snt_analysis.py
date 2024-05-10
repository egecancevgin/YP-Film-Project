import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import tf_keras as keras
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences
#from sklearn.utils import sequence
import pickle as pkl
from sklearn.model_selection import train_test_split
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
#from tf_keras.utils.np_utils import to_categorical



""" Calistirmadan once:
# apt update
# apt install python3
# apt install python3-pip
# pip install pandas
# pip install nltk
# pip install tensorflow
# pip install tf-keras
"""

nltk.download('stopwords')
STOPKELIMELERI = set(stopwords.words('english'))
TOPLAM_KELIME = 50000
KAYDEDILECEK_PATH = "./egitilmis_model.h5"
SEQ_UZUNLUGU = 250
EMB_BOYUTU = 100


def sutun_temizle(yorumlar):
    """ Review sutununa karsilik gelen degerlere ReGex ile on isleme yapar. """
    yorumlar = yorumlar.lower()
    yorumlar = ' '.join(kelime for kelime in yorumlar.split() if kelime not in STOPKELIMELERI)
    yorumlar = re.sub(r'\W', ' ', yorumlar)              # Özel karakterler
    yorumlar = re.sub(r'\s+[a-zA-Z]\s+', ' ', yorumlar)  # Tekli karakterler
    yorumlar = re.sub(r'\^[a-zA-Z]\s+', ' ', yorumlar)   # Cümle basi tekli karakterler
    yorumlar = re.sub(r'\s+', ' ', yorumlar, flags=re.I) # Çoklu whitespace'leri tekli yapalim
    return yorumlar


def verileri_oku(veri_yolu):
    """ Veriyi okur, DataFrame formatina cevirir ve on islemeler yapar. """
    veri = pd.read_csv(veri_yolu)
    # Verinin yarisi pozitif, yarisi negatif yorumlari iceriyor
    veri['review'] = veri['review'].apply(sutun_temizle)
    
    # Veriyi teker teker kelimelere ayırıp kullanılmayacak karakterleri filtreleyecek
    tokenizer = Tokenizer(num_words=TOPLAM_KELIME, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(veri['review'].values)
    
    # Veriyi bağımlı ve bağımsız değişkenlerine ayıralım
    X = tokenizer.texts_to_sequences(veri['review'].values)
    #X = pad_sequences(X, maxlen=LEN_OF_SEQ)
    X = pad_sequences(X, maxlen=SEQ_UZUNLUGU)

    with open('tokenizerTekrar','ab') as f:
        pkl.dump(tokenizer,f)

    Y = pd.get_dummies(veri['sentiment']).values
    return veri, X, Y


def modeli_egit(veri, X, Y):
    """ Modeli eğitir, değerlendirir ve döner. """
    # Önce %80'e-%20 olmak üzere eğitim-test ayrımını yapalım
    X_egitim, X_test, Y_egitim, Y_test = train_test_split(
        X, Y, test_size = 0.2, random_state = 42
    )

    # Bir Embedding + Dropout + LSTM + Dense katmanlı model oluşturalım
    model = Sequential()
    model.add(
        Embedding(TOPLAM_KELIME, EMB_BOYUTU, input_length = X.shape[1])
    )
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam', metrics=['accuracy']
    )

    # Modeli 5 tur ile eğitelim
    egitilmis_model = model.fit(
        X_egitim,
        Y_egitim, 
        epochs=5, 
        batch_size=64, 
        validation_split=0.1
    )

    # Modeli test edelim ve doğruluk sonucuna erişelim
    dogruluk = model.evaluate(X_test, Y_test)
    print(
        'Test seti:\n  Kayıp: {:0.3f}\n  Doğruluk: {:0.3f}'.format(
            dogruluk[0], dogruluk[1]
        )
    )
    return model



def main():
    """ Ana yurutucu ve test yapici fonksiyondur. """
    veri, X, Y = verileri_oku('./IMDB Dataset.csv')
    model = modeli_egit(veri, X, Y)
    model.save(KAYDEDILECEK_PATH)


if __name__ == '__main__':
    main()
