import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from szg_to_midi import pngs_to_numpy

# fix random seed for reproducibility
numpy.random.seed(7)

note_matrix = pngs_to_numpy('filename.png')

# normalize between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
note_matrix = scaler.fit_transform(note_matrix)

