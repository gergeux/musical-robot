# This file contains functions for converting a folder of MIDI files into SzG files.
# SzG binary arrays look like the following:
# [A0 Bb0 B0 C1 C#1 D1 ... A7 Bb7 B7 C8]
# The length of the array is 88, the values are always 0 (not pressed) or 1 (pressed)

import numpy as np
import mido
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from os import listdir
from os.path import isfile,join


note_matrix = [[0 for x in range(88)]]


def convert_append_note(note_dict_int, timetick, binary = False):
    """Converts a set of notes (integers) into a binary numpy array.
    Input argument is a dictionary of {note:velocity} pairs
    Empty dict or untouched note means no change
    """

    global note_matrix

    # if timetick >= 2**16 - 1:
    #     return

    while len(note_matrix) < timetick + 1:  # vertical line not there yet
        note_matrix.append(note_matrix[-1][:])  # REPEAT the last line no matter what

    if note_dict_int:  # not empty
        while note_dict_int['note'] < 21:
            note_dict_int['note'] += 12
        while note_dict_int['note'] > 87 + 21:
            note_dict_int['note'] -= 12

        if binary:
            if note_dict_int['velocity'] != 0:
                note_dict_int['velocity'] = 127

        if note_dict_int['velocity'] == 0:
            if timetick > 0:
                note_matrix[timetick - 1][note_dict_int['note'] - 21] = note_dict_int['velocity']  #  fontos mókolás!
                #  note_matrix[timetick][note_dict_int['note'] - 9] = note_dict_int['velocity']

        note_matrix[timetick][note_dict_int['note'] - 21] = note_dict_int['velocity']

    else:  # empty WHYYY
        note_matrix[timetick][:] = note_matrix[timetick - 1][:]
        pass

    #print(str(timetick) + '  ' + str(note_matrix[:][-1]))


def convert_file(file_name, undersampling_divider=1, binary=False):
    """Converts given file into glrglrbr
    Returns BMP-like matrix of midi file"""
    global note_matrix
    note_matrix = [[0 for x in range(88)]]
    mid = mido.MidiFile(file_name)

    print('tracks: ' + str(len(mid.tracks)))

    for trindex, track in enumerate(mid.tracks):
        in_tick_cnt = 0.0
        out_tick_cnt = 0

        for index, msg in enumerate(track):
            msg_dict = msg.dict()  # convert to dictionary
            #print(msg_dict)
            if not msg.is_meta :  # only consider non-meta messages
                # print(msg_dict)

                if msg_dict['type'] == 'note_off':
                    msg_dict['velocity'] = 0

                in_tick_cnt = in_tick_cnt + float(msg_dict['time'])/undersampling_divider
                # print(in_tick_cnt)

                if out_tick_cnt < in_tick_cnt:
                    out_tick_cnt += 1
                while out_tick_cnt < in_tick_cnt:
                    convert_append_note({}, out_tick_cnt, binary)  # everything stays the same, no event
                    out_tick_cnt += 1

                # if out_tick_cnt >= in_tick_cnt:
                if msg_dict['type'] == 'note_on' or msg_dict['type'] == 'note_off':  # don't send out controls
                    convert_append_note(msg_dict, out_tick_cnt, binary)  # new event, send data

            print('tr#' + str(trindex + 1) + ' ' + str(index) + '/' + str(len(track)))
        # print(in_tick_cnt)


def save_note_matrix(file_path):
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
    ix = 0
    for subimage in batch(note_matrix, 65535):
        plt.imsave(file_path + '_pt_' + str(ix+1) + '.png',
                   np.transpose(np.array(subimage)), format='png', origin='lower',
                   cmap=cm.gray, vmin=0, vmax=255)
        ix += 1


def convert_folder(folder_path, unders = 1, inclusive = False):
    """Converts a whole folder.
    Inclusive: Whether to include subfolders as well"""
    file_list = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]

    for file_path in file_list:
        convert_file(file_path, unders, binary=False)
        output_path = 'result_png/' + file_path.split('\\')[-1] + '_us_' + str(unders)
        save_note_matrix(output_path)

# midi_path = 'midi_data/Classical_mfiles.co.uk_MIDIRip/Waltz-op64-no2.mid'
# unders_div = 10
# convert_file(midi_path, undersampling_divider=unders_div, binary=False)
# print(len(note_matrix))
# ix = 0
# for subimage in batch(note_matrix, 65535):
#     plt.imsave('result_png/' + midi_path.split('/')[-1] + '_us_' + str(unders_div) + '_pt_' + str(ix+1) + '.png',
#             np.transpose(np.array(subimage)), cmap=cm.gray, format='png', origin='lower')
#     ix += 1

#convert_folder('midi_data\classical_piano\chopin', unders=4)

# convert_file('Waltz_OUTPUT_OP64_2.mid')
# save_note_matrix('Waltz-DOUBLE-no2.mid')

convert_file('midi_data/Classical_mfiles.co.uk_MIDIRip/Waltz-op64-no2.mid')
save_note_matrix('Waltz-op64-no2.mid')