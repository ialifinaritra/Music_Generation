import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pretty_midi
from absl import app

model_path = 'model'

sequence_length = 20
T_y_generated = 200


def random_sequence():
    notes = np.random.randint(0,79,20)
    X_ohe = np.zeros((20,79))
    for i, note in enumerate(notes):
        X_ohe[i, note] = 1

    return X_ohe


def generate_sequence(path, pattern):
    note_l = []
    prediction_l = []
    model = load_model(path)

    for note_index in range(T_y_generated):
        prediction = model.predict(np.expand_dims(pattern[note_index:,:], 0))
        note_pred = np.argmax(prediction)
        note_l.append(note_pred)

        one_hot = np.zeros((1, 79))
        one_hot[0, note_pred] = 1

        prediction_l.append(one_hot)

        pattern = np.vstack((pattern, one_hot))

    return note_l, prediction_l


def create_audio(notes):
    new_midi_data = pretty_midi.PrettyMIDI()
    cello_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    cello = pretty_midi.Instrument(program=cello_program)
    time = 0
    step = 0.3

    for note_number in notes:
        myNote = pretty_midi.Note(velocity=100, pitch=note_number, start=time, end=time+step)
        cello.notes.append(myNote)
        time += step

    new_midi_data.instruments.append(cello)

    return new_midi_data


def main(_):
    input = random_sequence()
    note_l, pred_l = generate_sequence(model_path, input)
    pred_audio = create_audio(note_l)
    pred_audio.write('pred_music.mid')


if __name__ == '__main__':
    app.run(main)
