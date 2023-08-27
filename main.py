import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load dataset
df = pd.read_csv('md_chat_dataset.csv')

# Split the dataset into input and target
input_texts = df['message']
target_texts = df['message'].shift(-1).fillna('')

# Split into train and test sets
input_train, input_test, target_train, target_test = train_test_split(input_texts, target_texts, test_size=0.2, random_state=42)


# Tokenization
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(input_train.astype(str))
input_sequences = tokenizer.texts_to_sequences(input_train.astype(str))
target_sequences = tokenizer.texts_to_sequences(target_train)

# Padding
max_sequence_length = max(max(len(seq) for seq in input_sequences), max(len(seq) for seq in target_sequences))
input_sequences_padded = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
target_sequences_padded = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 128, input_length=max_sequence_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
    tf.keras.layers.Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import os

model_path = 'trained_model.h5'  # Path to save/load the trained model


def chat_with_model(input_text):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_sequence_padded = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='post', truncating='post')
    predicted_sequence = model.predict(input_sequence_padded)[0]
    predicted_text = tokenizer.sequences_to_texts([[np.argmax(token_probs) for token_probs in predicted_sequence]])[0]
    return predicted_text

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Loaded trained model.")
    user_input = "thanks"
    response = chat_with_model(user_input)
    print("Chatbot:", response)
else:
    # Train the model
    model.fit(input_sequences_padded, target_sequences_padded, epochs=10, verbose=1)
    
    # Save the trained model
    model.save(model_path)
    print("Trained model saved.")