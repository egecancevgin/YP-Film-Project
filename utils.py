from pathlib import Path
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re
import tensorflow as tf
from nltk.corpus import stopwords


STOPWORDS = set(stopwords.words('english'))

PATH = Path(__file__).parent.parent
EMB_DIM = 100
TOTAL_WORDS = 50000
LEN_OF_SEQ = 250

def clean_review(review):
    review = review.lower() # küçük harf
    review = ' '.join(word for word in review.split() if word not in STOPWORDS) # review'dan stopwordsleri kaldır
    review = re.sub(r'\W', ' ', review) # özel karakterleri kaldır
    review = re.sub(r'\s+[a-zA-Z]\s+', ' ', review)  # tekli karakterleri kaldır
    review = re.sub(r'\^[a-zA-Z]\s+', ' ', review) # cümle başındaki tekli karakterleri kaldır
    review = re.sub(r'\s+', ' ', review, flags=re.I) # çoklu boşlukları tekli yap
    return review



def get_tokenizer():
    with open(str(PATH)+"/tokenizerPickle","rb") as f:
        tokenizer  = pickle.load(f)
    
    return tokenizer


def review_data_pipeline(review:str)->np.ndarray:
    tokenizer = get_tokenizer()
    review = clean_review(review)
    review = tokenizer.texts_to_sequences([review])
    review = pad_sequences(review,maxlen=LEN_OF_SEQ)
    return review



def get_model():
    model = tf.keras.models.load_model(str(PATH)+"/checkpoint/final_modelh5.h5")
    return model
