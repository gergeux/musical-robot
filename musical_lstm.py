# 1. Read a folder (?) of MIDI files into one concatenated numpy array. (Tx88)
#   Track separator should be the deepest note or all notes or something.
# 2. Cut the data into chunks of length L timeticks with step S.
#   Reshape 'input' array to size (D, L, 88).
#   Where D is the total number of chunks (sentences): D ~= (T-L)/S
# 3. Also create array 'y' for storing next note row that comes after each D sentence.
#   size(y) = (D, 88)
# 4. Now we have input matrix X and output 'vectors' y for training the network.
#   Save the model after every 10th (?) iteration.
#   Generate an output midi file (of fixed timetick length) every 10th (?) iteration.

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils.data_utils import get_file
from midi_to_szg import midi_to_numpy
import numpy as np
from matplotlib import pyplot as plt
import gc
import random
import sys

# load one midi file (for now), binary mode (for now)
midi_file = 'midi_data/Classical_mfiles.co.uk_MIDIRip/Waltz-op64-no2.mid'
note_matrix = midi_to_numpy(midi_file, binary=True, undersampling_divider=4)

print('len:', len(note_matrix))  # should ouput the length of the midi in timeticks

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 100  # number of notes in one sentence. I have no how many it should be
step = 10  # this is similar to undersampling but from a different perspective
sentences = []
next_notes = []
for i in range(0, len(note_matrix) - maxlen, step):
    sentences.append(note_matrix[i: i + maxlen])
    next_notes.append(note_matrix[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
print('x size: ' + str(len(sentences)) + '*' + str(maxlen) + '*' + str(len(note_matrix[0])))
X = np.zeros((len(sentences), maxlen, len(note_matrix[0])), dtype=np.bool)  # type=bool is ONLY for binary
y = np.zeros((len(sentences), len(note_matrix[0])), dtype=np.bool)  # type=bool is ONLY for binary
# have to completely rewrite this following loop:
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, :] = char
    y[i, :] = next_notes[i]

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(note_matrix[0]))))
model.add(Dense(len(note_matrix[0])))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# def sample(preds, temperature=1.0):
#     # helper function to sample an index from a probability array
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)


# train the model, output generated text after each iteration
for iteration in range(1, 60):
    # training part
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, epochs=1,
              callbacks=[ModelCheckpoint('lstm_states/weights.{epoch:02d}.hdf5')])

    # model_state_path = 'lstm_states/' + 'lstm_model_epoch_' + str(iteration) + '.lstm'
    # model.save(model_state_path)

import gc

gc.collect()  # collect garbage - necessary due to tensorflow error
