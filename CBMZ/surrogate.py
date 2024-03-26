import os

import tensorflow.compat.v1 as tf

from src.build_nn import build_model
from src.surrogate import Surrogate, batched

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Hyperparameter Tuning (mkelp)
num_blocks = 2
num_layers = 4
latent_size = 256
compressed_species = 16
dropout = 0.35
runnum = 1
n_steps = 3
n_conc = 101

class KelpNN(Surrogate):
    def __init__(self, model_name=None):
        diurnal_model, self.integrator, self.encoder, self.decoder = build_model(num_blocks, num_layers, latent_size, compressed_species, dropout, runnum, n_steps)
        if model_name is not None:
            self.load_weights(model_name)
        super().__init__(diurnal_model)

    def load_weights(self, model_name):
        self.integrator.load_weights(model_name.format('Model'))
        self.encoder.load_weights(model_name.format('Encoder'))
        self.decoder.load_weights(model_name.format('Decoder'))

    def fit(self, X, y):
        pass

    @batched(int(2.5e4))
    def predict(self, inputs):
        X, met = inputs
        X = tf.expand_dims(X, 1)
        noise = tf.zeros((X.shape[0], n_steps-1, n_conc), dtype=X.dtype)
        X = tf.concat([X, noise], 1)
        full_pred = self.model.predict(x=[X, met, noise], steps=1)[-1]
        y_preds = full_pred[:,-1,10]

        return y_preds
