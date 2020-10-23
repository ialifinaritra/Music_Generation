import os
import pretty_midi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten, Dropout, Activation
import numpy as np
from absl import app, flags
import glob
import tensorflow as tf

n_x = 79
max_T_x = 1000
sequence_length = 20
T_y_generated = 200

NB_EPOCHS = 30
PATH_FILE = 'data/'
OUTPUT_PATH = 'model/'


def read_convert_data(data_path):
    list_file = glob.glob(data_path + 'cs*.mid')
    X_list = []

    for file in list_file:
        midi_data = pretty_midi.PrettyMIDI(file)
        note_l = [note.pitch for note in midi_data.instruments[0].notes]

        # convert to one-hot-encoding
        T_x = len(note_l)
        if T_x > max_T_x:
            T_x = max_T_x
        X_ohe = np.zeros((T_x, n_x))

        for t in range(T_x):
            X_ohe[t, note_l[t] - 1] = 1

        X_list.append(X_ohe)

    return X_list


def training(nb_epoch, data_path, output_path):
    X_list = read_convert_data(data_path)
    x_train = []
    y_train = []

    for i in range(len(X_list)):
        for t in range(X_list[i].shape[0] - sequence_length):
            x_train.append(X_list[i][t: t + sequence_length, :])
            y_train.append(X_list[i][t + sequence_length, :])

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    # Create the model
    model = Sequential()
    model.add(LSTM(256,input_shape = (20,79),return_sequences=True))
    model.add(Dropout(rate = 0.3))

    model.add(LSTM(256,return_sequences = True ))
    model.add(Dropout(rate = 0.3))

    model.add(LSTM(256,return_sequences = False))

    model.add(Dense(256,activation = 'relu'))
    model.add(Dropout(rate = 0.3))

    model.add(Dense(79, activation='softmax'))

    print(model.summary())

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_path,
        save_weights_only=False,
        save_best_only=True,
        monitor='accuracy',
        mode='max'
    )

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=nb_epoch, batch_size=64, verbose=1, callbacks=[checkpoint_cb])


def main(_):
    training(NB_EPOCHS, PATH_FILE, OUTPUT_PATH)


if __name__ == '__main__':
    app.run(main)