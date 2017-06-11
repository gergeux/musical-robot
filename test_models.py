### this is just a testing environment for testing a model trained on op64no2
### it loads all the models in lstm_states and generates output midi file
### the seed for each model is the first sentence of the actual midi track

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from sklearn.preprocessing import normalize
from keras.utils.data_utils import get_file
from os import listdir
from os.path import isfile, join
from midi_to_szg import midi_to_numpy
from midi_to_szg import save_note_matrix
from szg_to_midi import numpy_to_midi
import numpy as np
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



# test the model, output generated text after each iteration
file_list = [join('lstm_states_old', f) for f in listdir('lstm_states_old') if isfile(join('lstm_states_old', f))]
for iteration, file_path in enumerate(file_list):
    # testing part
    note_out_matrix = np.zeros((1, 88), 'int')  # for holding the output
    print()
    print('-' * 50)
    print('Iteration', iteration)
    print(file_path)

    model.load_weights(file_path)

    # output generating part -- will think about this later
    #start_index = random.randint(0, len(text) - maxlen - 1)

    # for diversity in [0.2, 0.5, 1.0, 1.2]:
    if True:
        # print()
        # print('----- diversity:', diversity)

        # generated = ''
        # sentence = text[start_index: start_index + maxlen]
        # generated += sentence
        # print('----- Generating with seed: "' + sentence + '"')
        # sys.stdout.write(generated)

        sentence = sentences[480]
        print(sentence[0])
        note_out_matrix = np.vstack((note_out_matrix, sentence))

        for i in range(4000): # length should be 65535 - about a minute or so
            # print('Predicting note #' + str(i))
            # preds = model.predict(sentences[0], verbose=0)[0] * 127
            preds = model.predict(np.reshape(sentence, (1, len(sentence), len(sentence[0]))), verbose=0)[0]
            # print(np.min(preds))
            # print(np.ptp(preds))
            preds = (preds - np.min(preds))/np.ptp(preds)  # normalization
            preds *= 127  # velocity values
            preds = np.asarray(preds).astype('int')
            # print(preds)
            # print(type(preds))
            # print(len(preds))

            # print(np.shape(note_out_matrix))
            note_out_matrix = np.vstack((note_out_matrix, preds))
            # print(np.shape(note_out_matrix))
            # print(np.shape(sentence))
            sentence = np.vstack((sentence[1:][:], preds))
            # print(np.shape(sentence))
        # print()

    print(np.shape(note_out_matrix))
    print(note_out_matrix[125])
    numpy_to_midi(note_out_matrix, 'lstm_states_old/epoch_' + str(iteration+1) + '.mid')
    # save_note_matrix('rohatteletbebakker', note_out_matrix)  # just for the image

# midi_to_numpy('output1_remenykedek.mid')
# save_note_matrix('gecispicsabapng')

import gc
gc.collect()  # collect garbage - necessary due to tensorflow error
