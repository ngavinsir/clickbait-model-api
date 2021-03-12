from fastapi import FastAPI, Body
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def load_chosen_dataset_input(input_file_path, random_state=10284):
    df = pd.read_csv(input_file_path, sep=',', header=0)

    label_names=['label_score']
    headline = df['title']
    label = df[label_names].values

    X_train, X_test, y_train, y_test = train_test_split(
            headline, label, stratify=label, test_size=0.2, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

def preprocess_dataset(X_train, X_test):
    raw_docs_train = X_train
    raw_docs_test = X_test
    tokenizer = RegexpTokenizer(r'\w+')
    max_seq_len = 20
    num_classes = 1

    processed_docs_train = []
    for doc in raw_docs_train:
        tokens = tokenizer.tokenize(doc)
        filtered = [word for word in tokens]
        processed_docs_train.append(" ".join(filtered))

    processed_docs_test = []
    for doc in raw_docs_test:
        tokens = tokenizer.tokenize(doc)
        filtered = [word for word in tokens]
        processed_docs_test.append(" ".join(filtered))
        
    return processed_docs_train, processed_docs_test

def create_word_index_dict(all_processed_data):
    tokenizer = Tokenizer(num_words=30000, lower=True, char_level=False)
    tokenizer.fit_on_texts(all_processed_data)
    word_index = tokenizer.word_index
    
    return tokenizer

model = load_model(
    './model_no_symbol_rnn_2.h5',
    custom_objects={'f1_m':f1_m, 'precision_m':precision_m, 'recall_m':recall_m}
)
X_train, X_test, y_train, y_test = load_chosen_dataset_input('./fa_no_symbol.csv')
train_data_processed, test_data_processed = preprocess_dataset(X_train, X_test)
tokenizer = create_word_index_dict(train_data_processed+test_data_processed)

app = FastAPI()

@app.post("/predict_clickbait")
def predict_clickbait(headline: str = Body(...)):
    input = tokenizer.texts_to_sequences([headline])
    return True if K.get_value(model(np.array(input)))[0,0] >= 0.5 else False