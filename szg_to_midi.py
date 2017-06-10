from mido import Message, MidiFile, MidiTrack
from scipy import misc
import numpy as np
import os.path


def pngs_to_numpy(file_png):
    # Concatenates png part files starting from file_png into a numpy array
    # Returns note_matrix numpy array
    note_matrix = np.zeros((1, 88), 'int')
    part_no = 1
    file_png_part = (file_png.split('\\')[-1])[:-5] + str(part_no) + '.png'  # 'Waltz_op_no2.mid_pt_' + '1' + '.png'

    while os.path.isfile(file_png_part):
        print(file_png_part)
        note_matrix = np.vstack((note_matrix, np.ndarray.astype(np.flip(np.transpose(misc.imread(file_png_part)[:, :, 0]), 1), 'int')))
        part_no += 1
        file_png_part = (file_png.split('\\')[-1])[:-5] + str(part_no) + '.png'  # 'Waltz_op_no2.mid_pt_' + '1'

    return note_matrix


def numpy_to_midi(note_matrix, file_mid):
    # Converts and saves note_matrix into file_mid midi file
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(Message('program_change', program=0, time=0))

    last_msg_timetick = 0
    timetick_differential = 0
    for timetick, time_row in enumerate(note_matrix):
        if timetick == 0:
            # print(time_row)
            for note, velocity in enumerate(time_row):
                if time_row[note] != 0:
                    # print('t: ' + str(timetick) + ', d: ' + str(timetick_differential) +
                    #       ', note: ' + str(note + 21) + ', vel: ' + str(time_row[note]))

                    track.append(
                        Message('note_on', note=note + 21, velocity=time_row[note], time=timetick_differential))

        if timetick > 0:
            note_diff = time_row - note_matrix[timetick - 1]

            if np.count_nonzero(note_diff) > 0:  # there was a change
                for note, velocity in enumerate(time_row):
                    if note_diff[note] != 0:
                        timetick_differential = timetick - last_msg_timetick
                        # print('t: ' + str(timetick) + ', d: ' + str(timetick_differential) + ', note: ' + str(
                        #     note + 21) + ', vel: ' + str(time_row[note]))

                        track.append(
                            Message('note_on', note=note + 21, velocity=time_row[note], time=timetick_differential))
                        last_msg_timetick = timetick

    mid.save(file_mid)  # save and close midi file


def pngs_to_midi(file_png, file_mid):
    # Wrapper Concatenates png part files starting from file_png into a numpy array
    # then converts numpy array into midi track and saves into file_mid
    note_matrix = pngs_to_numpy(file_png)
    numpy_to_midi(note_matrix, file_mid)


pngs_to_midi('Waltz-op64-no2.mid_pt_1.png', 'Waltz_OUTPUT_OP64_2.mid')