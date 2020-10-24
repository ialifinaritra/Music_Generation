<h1 align='center'> Sebastian Bach's Music Sequences Generation </h1> 

# Introduction
This repository contains the code for training a network to learn a language model and then use the model to generate new sequences. The model will be trained to learn the language of music of [Johann_Sebastian_Bach](https://en.wikipedia.org/wiki/Johann_Sebastian_Bach).

For this, the model will learn how J.S Bach's "Cello suite" have been composed. 

For sparsing music file , a python package named `PrettyMidi` need to be installed. This code is tested on `TF 2.3` .


# Training 

The traning will be done on the whole set MIDI files of the "Cello suites". 36 midi files are used for training the model.

Each music is sampled into suit of 20 notes and forward to LSTM neural network.

# Generation

For the generation step, 20 random notes will be given to the model and it will generate a new Cello suits containing `nb_generated` notes. 

This example will generate a suit of 300 notes and create the midi file into `pred_music.mid`.

<pre><code> python generate_music.py --nb_generated=300
</code></pre>
