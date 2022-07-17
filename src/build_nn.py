from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import tensorflow.compat.v1 as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

from config import CBMZ_inputs

substances = CBMZ_inputs["labels"]
met_names = CBMZ_inputs["met_labels"]

def make_mlp_model(latent_size=16, num_layers=2, dropout=0.5, name="MLP"):
    """Instantiates a new MLP followed by BatchNorm and Dropout as a Keras makel

    Args:
      latent_size: The number of nodes in each layer
      num_layers: The number of layers
      dropout: Fraction of the input units to drop in the dropout layer.

    Returns:
      The MLP building function
    """
    model = tf.keras.Sequential(name=name)
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(latent_size, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())

        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))

    return model

def make_codec(output_size, name="codec"):
    """ Creates a new linear encoder or decoder with the given number of
      output units.
    """
    model = tf.keras.Sequential(name=name)

    model.add(tf.keras.layers.Dense(units=output_size, activation=None, name="codec_layer"))
    return model

def resnet(n_conc,
          n_met,
          model_fn=make_mlp_model,
          res_blocks=2,
          name="ResNet"):
    """
    A residual MLP neural network building function.
      Args:
        n_conc: Number of concentration inputs
        n_met: Number of meteorology inputs
        model_fn: a trainable model block
        residual: the number of times the model_fn should be repeated
          with residual skips in between.
    """
    conc_inputs = tf.keras.layers.Input(shape=(n_conc,), name="conc_input")
    met_inputs = tf.keras.layers.Input(shape=(n_met,), name="met_input")

    conc = conc_inputs
    outputs = []
    for i in range(res_blocks):
        inputs = tf.keras.layers.concatenate([conc, conc_inputs, met_inputs], axis=-1)
        core = model_fn("MLP_block_%d"%i)
        output_transform = tf.keras.layers.Dense(conc_inputs.shape[-1], name="output_transform_%d"%i)
        transformed_output = output_transform(core(inputs))
        conc = tf.keras.layers.add([transformed_output, conc_inputs], name="skip_%d"%i)
        outputs.append(conc)

    model = tf.keras.Model(inputs=[conc_inputs, met_inputs], outputs=outputs, name=name)
    model.n_blocks = res_blocks
    return model

def add_noise(inputs, step):
    [outputs, noise] = inputs
    for i, o in enumerate(outputs):
        outputs[i] = tf.keras.layers.add([o, noise], name="add_noise_s%d_b%d"%(step,i))
    return outputs

def stack_outputs(out, n_blocks):
    o_temp = {}
    for i in range(len(out)):
        block = i%n_blocks
        if i<n_blocks: o_temp[i] = []
        o_temp[block].append(out[i])
    outputs = []
    for i in range(len(o_temp)):
        outputs.append(tf.stack(o_temp[i], axis=1))
    return outputs

def roll_out_model(model, encoder, decoder, n_conc, n_met, n_steps, name="roll_out"):
    """ Encodes, the input data, runs the given model through a number of steps,
       using the output of each
       step as the input of the next step, and returns the decoded
       output of all steps. Results are not decoded between steps.

       Args:
        model: Keras model to run for each step
        encoder: Keras model to encode the input to each step.
        decoder: Keras model to decode the output from each step.
        n_conc: Number of concentration variables
        n_met: number of meteorology variables
        n_steps: Number of steps to integrate through
    """
    conc_input = tf.keras.layers.Input(shape=(n_steps+1, n_conc,), name="rollout_conc_input")
    met = tf.keras.layers.Input(shape=(n_steps+1, n_met,), name="rollout_met")
    noise = tf.keras.layers.Input(shape=(n_steps, n_conc,), name="rollout_noise")

    # First step of input concentration
    conc = tf.keras.layers.Lambda(lambda inputs: inputs[:,0,:], name="conc_subset")(conc_input)
    conc = encoder(conc)

    out_temp = []
    for i in range(n_steps):
        subset = tf.keras.layers.Lambda(lambda inputs: inputs[:,i,:], name="subset_%d"%i)
        outputs = model([conc, subset(met)])
        outputs = tf.keras.layers.Lambda(lambda x: add_noise(x, i), name="step_%d"%i)(
                                                  [outputs,encoder(subset(noise))])
        conc = outputs[-1]
        for o in outputs: out_temp.append(decoder(o))
        n_blocks = len(outputs)

    out = tf.keras.layers.Lambda(lambda x: stack_outputs(x, n_blocks), name="stack_layer")(out_temp)

    return tf.keras.Model(inputs=[conc_input,met,noise], outputs=out, name=name)
    

def o3(d): return d[...,10:11]

def mse_focus_o3(y_true, y_pred, focus_factor=10000):
  """ A loss metric that priorizes PM accuracy by the specified factor. """
  y_pred_o3 = tf.concat([y_pred, o3(y_pred)*focus_factor],axis=-1)
  y_true_o3 = tf.concat([y_true, o3(y_true)*focus_factor],axis=-1)
  return K.mean(tf.math.square(y_pred_o3 - y_true_o3), axis=-1)

#Hyperparameter Tuning (mkelp)
num_blocks = 2
num_layers = 4
latent_size = 256
compressed_species = 16
dropout = 0.35
runnum = 1
n_steps = 24
batch_size_tr = 1024
num_rec_tr = 500000*8
n_steps_tr = int(num_rec_tr/batch_size_tr*(10.0/12.0))
n_steps_eval = int(num_rec_tr/12.0/batch_size_tr)


def build_model(num_blocks, num_layers, latent_size, compressed_species, dropout, runnum, n_steps=n_steps):
    model = resnet(n_conc=compressed_species, n_met=len(met_names),res_blocks=num_blocks,
                   model_fn=lambda name: make_mlp_model(latent_size=latent_size,
                   num_layers=num_layers, dropout=dropout, name=name))

    encoder = make_codec(compressed_species, name="encoder")
    decoder = make_codec(len(substances), name="decoder")

    diurnal_model = roll_out_model(model, encoder, decoder, len(substances), len(met_names), n_steps-1, name="diurnal_roll_out")

    print("print LR", batch_size_tr / 8.0 * 1.0e-5 )
    print("print eval steps", n_steps_eval)
    print("print steps", int(num_rec_tr/batch_size_tr*(10.0/12.0)))

    print("NUMBER OF COMPRESSED SPECIES: ", compressed_species)

    learning_rate = batch_size_tr / 8.0 * 1.0e-5
    diurnal_model.compile(
            tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=mse_focus_o3,
            metrics=['mean_squared_error', mse_focus_o3])

    return diurnal_model, model, encoder, decoder
