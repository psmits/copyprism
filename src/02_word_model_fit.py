from numpy import array
from pickle import dump
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from copyprisim_utilities import load_doc

# load
in_filename = 'ikea_word_train_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')

print('total sequences: %d' % len(lines))
# code as integers

# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
dump(tokenizer, open('word_tokenizer.pkl', 'wb'))

# vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# separate into input and output
sequences = array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

# define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=seq_length))
model.add(LSTM(100))  # , return_sequences=True))
# You must set return_sequences=True when stacking LSTM layers so that the
# second LSTM layer has a three-dimensional sequence input
# model.add(Dropout(0.1))
# model.add(LSTM(100))
# model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
# print(model.summary())

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# helpful checkpoints
filepath = "word_model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='loss',  # 'accuracy'
                             verbose=1,
                             save_best_only=True,
                             mode='min')
desired_callbacks = [checkpoint]

# fit model
model.fit(X, y, epochs=100, batch_size=256, callbacks=desired_callbacks)

# save the model
model.save('ikea_word_model.h5')
# save the tokenizer
dump(tokenizer, open('word_tokenizer.pkl', 'wb'))
