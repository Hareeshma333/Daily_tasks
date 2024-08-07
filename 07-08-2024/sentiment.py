# -*- coding: utf-8 -*-
"""Sentiment.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1U5so7fUfGvCgO7pModIDn8Y6jM0xkqLa
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#sample data
sentences =[
    "i love my bike!",
    "i had a  worst experience in service",
    "it was ok, but nothing special",
    "the service consultant was terrible"
]
#positive =1, negative =0, neutral=2
labels =[1,0,2,0]

#tokenization and the padding of sequences

tokenizer = Tokenizer(num_words=10000,oov_token='<oov>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

#sequences
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')

#create the model for the above
model = Sequential(
    [
        Embedding(input_dim=10000, output_dim=64),
        LSTM(64,return_sequences = False),
        Dropout(0.5),
        Dense(3,activation='softmax')

    ]
)
#compilation can be done for the model
model.compile(loss = 'sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#train the model
model.fit(padded,np.array(labels),epochs=100,verbose=1)

#make the prediction
test_sentences = ["i enjoy the filim"]
test_seq = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_seq, maxlen=padded.shape[1])
prediction = model.predict(test_padded)
predicted_class =np.argmax(prediction)
print(f"predicted sentiment of the sentence : {predicted_class}")