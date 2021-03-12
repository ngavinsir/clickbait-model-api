from tensorflow.keras import backend as K, optimizers
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import os, re, csv, math, codecs

adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

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

metrics = ['accuracy', f1_m, precision_m, recall_m]

def get_model_bilstm(metrics_input):
    model_bilstm = Sequential()
    model_bilstm.add(Embedding(nb_words, 300,
              weights=[embedding_matrix], input_length=20, trainable=False))
#     model_bilstm.add(Embedding(max_features, 128))
    model_bilstm.add(Bidirectional(LSTM(128)))
    model_bilstm.add(Dropout(0.5))
    model_bilstm.add(Dense(1, activation='sigmoid'))

    model_bilstm.compile(loss='binary_crossentropy', optimizer=adam, metrics=metrics_input)
    return model_bilstm

def load_chosen_dataset_input(input_file_path, random_state=10284):
    df = pd.read_csv(input_file_path, sep=',', header=0)

    label_names=['label_score']
    headline = df['title']
    label = df[label_names].values

    X_train, X_test, y_train, y_test = train_test_split(
            headline, label, stratify=label, test_size=0.2, random_state=random_state)

    # print("total: ", df['title'].shape[0])
    # print("train: ", X_train.shape[0])
    # print("test: ", X_test.shape[0])
    
    return X_train, X_test, y_train, y_test

def preprocess_dataset(X_train, X_test):
    raw_docs_train = X_train
    raw_docs_test = X_test
    tokenizer = RegexpTokenizer(r'\w+')
    max_seq_len = 20
    num_classes = 1

    # print("pre-processing train data...")
    processed_docs_train = []
    for doc in tqdm(raw_docs_train):
        tokens = tokenizer.tokenize(doc)
        filtered = [word for word in tokens]
        processed_docs_train.append(" ".join(filtered))

    processed_docs_test = []
    for doc in tqdm(raw_docs_test):
        tokens = tokenizer.tokenize(doc)
        filtered = [word for word in tokens]
        processed_docs_test.append(" ".join(filtered))
        
    return processed_docs_train, processed_docs_test

def tokenize_input(train_data_processed, test_data_processed, tokenizer):
    # print("Tokenizing Input Data")
    # print("-"*30)
    # print("Fit all existing words (assigning index)...")

    # print("\nTransform input data into a sequence of integers(index)..")
    train_data_sequenced = tokenizer.texts_to_sequences(train_data_processed)
    test_data_sequenced = tokenizer.texts_to_sequences(test_data_processed)
    
    return train_data_sequenced, test_data_sequenced
    

def pad_sequences(train_data_sequenced, test_data_sequenced, max_seq_len):
    # print("padding sequences to have same length...")
    train_data_padded = sequence.pad_sequences(train_data_sequenced, maxlen=max_seq_len)
    test_data_padded = sequence.pad_sequences(test_data_sequenced, maxlen=max_seq_len)
    
    return train_data_padded, test_data_padded

def create_word_index_dict(all_processed_data):
    tokenizer = Tokenizer(num_words=30000, lower=True, char_level=False)
    tokenizer.fit_on_texts(all_processed_data)
    word_index = tokenizer.word_index
    # print("dictionary size: ", len(word_index))
    
    return tokenizer

def load_indonesian_word_embeddings():
    print('loading indonesian word embeddings...')
    indonesian_embeddings_index = {}
    fasttext_indo = codecs.open(
        '/media/ngavinsir/DATA/Project/cc/cc.id.300.vec', encoding='utf-8'
    )
    for line in tqdm(fasttext_indo):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        indonesian_embeddings_index[word] = coefs

    fasttext_indo.close()
    print('found %s word vectors' % len(indonesian_embeddings_index))
    
    return indonesian_embeddings_index

def plot_embedding_matrix(word_index, embeddings_index):
    embed_dim = 300
    words_not_found = []
    
    # print('preparing embedding matrix...')

    nb_words = min(30000, len(word_index)+1)
    
    embedding_matrix = np.zeros((nb_words, embed_dim))

    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

        else:
            words_not_found.append(word)
    # print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    # print("sample words not found: ", np.random.choice(words_not_found, 10))
    
    return embedding_matrix, nb_words

model = load_model('./model_no_symbol_rnn_2.h5',
                   custom_objects={'f1_m':f1_m, 'precision_m':precision_m, 'recall_m':recall_m, 'f1_m':f1_m})
X_train, X_test, y_train, y_test = load_chosen_dataset_input('./fa_no_symbol.csv')
train_data_processed, test_data_processed = preprocess_dataset(X_train, X_test)
tokenizer = create_word_index_dict(train_data_processed+test_data_processed)
# word_index = tokenizer.word_index
# embedding_index = load_indonesian_word_embeddings()
# embedding_matrix, nb_words = plot_embedding_matrix(word_index, embedding_index)
# model = get_model_bilstm(metrics)

train_data_sequenced, test_data_sequenced = tokenize_input(
    train_data_processed, 
    test_data_processed, 
    tokenizer
)
train_data, test_data = pad_sequences(train_data_sequenced, test_data_sequenced, 20)
input = tokenizer.texts_to_sequences(['menkum ham sudah kaji draf revisi uu kpk'])
print(True if K.get_value(model(np.array(input)))[0,0] >= 0.5 else False)
# model_train = model.fit(train_data, y_train, batch_size=256, epochs=7, verbose=2)
# model.save('./model_no_symbol_rnn_2.h5')