import csv, os
import numpy as np
import pandas as pd
import tensorflow as tf
from model.autoencoder import Autoencoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
from model.utils import *

def callbacks(config, initial_epoch  = 0):

    checkpoint_filepath = config.model_path
    log_file = config.log_path

    if os.path.exists(checkpoint_filepath):
        print("Checkpoint found. Loading weights.")
        Autoencoder.load_weights(checkpoint_filepath)
    else:
        print("No checkpoint found. Starting training from scratch.")

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    class CSVLoggerCallback(tf.keras.callbacks.Callback):
        def __init__(self, filename):
            super(CSVLoggerCallback, self).__init__()
            self.filename = filename

            if not os.path.exists(self.filename):
                with open(self.filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['epoch', 'loss', 'val_loss'])

            self.start_epoch = 0

            if os.path.exists(self.filename):
                with open(self.filename, mode='r') as file:
                    reader = csv.reader(file)
                    lines = list(reader)
                    if len(lines) > 1:
                        self.start_epoch = int(lines[-1][0]) + 1

        def on_epoch_end(self, epoch, logs=None):
            with open(self.filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + self.start_epoch, logs.get('loss'), logs.get('val_loss')])

    csv_logger_callback = CSVLoggerCallback(log_file)

    initial_epoch = csv_logger_callback.start_epoch

    return initial_epoch, model_checkpoint_callback, csv_logger_callback

def configure_lr(config):
    initial_learning_rate = config.learning_rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=config.decay_steps,
        decay_rate=config.decay_rate,
        staircase=config.staircase
    )

    optimizer = Adam(learning_rate=lr_schedule)
    Autoencoder.compile(optimizer, loss=MSE())
    return Autoencoder


def evaluation(Autoencoder, test_data,config):
    Autoencoder.load_weights(config.model_path)
    return compute_nmse_db(test_data, Autoencoder, awgn_riccian)


def compute_nmse_db(test_ori, Autoencoder, awgn):
    snr_range = np.arange(0, 16, 1)
    nmse_db_values = []

    Encoder = Autoencoder.enc.predict(x=test_ori, batch_size=1000)
    for snr_db in snr_range:
        noise = awgn(Encoder, snr_db)
        output = Autoencoder.dec.predict(x=noise, batch_size=1000)
        nmse = np.power(np.linalg.norm(test_ori - output), 2) / np.power(np.linalg.norm(test_ori), 2)
        nmse_db = 10 * np.log10(nmse)
        nmse_db_values.append(nmse_db)
        print(f"SNR: {snr_db} dB, NMSE: {nmse}, NMSE(dB): {nmse_db}")

    nmsedb_vs_snrdb = pd.DataFrame({
        'SNR(dB)': snr_range,
        'NMSE(dB)': nmse_db_values
    })

    return nmsedb_vs_snrdb



