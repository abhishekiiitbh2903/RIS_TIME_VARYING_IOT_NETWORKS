import tensorflow as tf
import numpy as np
from scipy.stats import rice
from tensorflow.keras.layers import Conv1D,Layer,Activation
from tensorflow.keras.layers import Add



class DRblock(Layer):
    def __init__(self, neurons):
        super().__init__()
        self.convs = [Conv1D(neurons, 3, padding='same', activation='relu') for _ in range(3)]
        self.conv_final = Conv1D(neurons, 3, padding='same')

    def call(self, input):
        output = input
        for conv in self.convs:
            output = conv(output)
        output = self.conv_final(output)
        output = Add()([output, input])
        output = Activation('relu')(output)

        return output
    

@tf.autograph.experimental.do_not_convert
def awgn_riccian(input_signal, snr_dB, rate=1.0, K=3):
    snr_linear = 10 ** (snr_dB / 10.0)
    b = np.sqrt(K / (K + 1))
    
    output = tf.TensorArray(tf.float32, size=input_signal.shape[0])

    for i in range(input_signal.shape[0]):
        input = input_signal[i]

        h_real = rice.rvs(b, size=input.shape).astype(np.float32)
        h_imag = rice.rvs(b, size=input.shape).astype(np.float32)

        faded_signal_real = h_real * input
        faded_signal_imag = h_imag * input

        avg_energy = tf.reduce_mean(tf.abs(input) ** 2)
        avg_energy = tf.maximum(avg_energy, 1e-8)
        noise_variance = avg_energy / snr_linear
        noise_real = tf.random.normal(input.shape, dtype=tf.float32)
        noise_imag = tf.random.normal(input.shape, dtype=tf.float32)
        noise_real = tf.sqrt(noise_variance / 2.0) * noise_real
        noise_imag = tf.sqrt(noise_variance / 2.0) * noise_imag
        output_signal_real = faded_signal_real + noise_real
        output_signal_imag = faded_signal_imag + noise_imag
        magnitude_signal = tf.sqrt(output_signal_real**2 + output_signal_imag**2)  
        output = output.write(i, tf.expand_dims(magnitude_signal, axis=0))
    output = output.stack()
    output = tf.reshape(output, [output.shape[0], 1, input_signal.shape[2]])
    return output


@tf.autograph.experimental.do_not_convert
def awgn(input_signal, snr_dB, rate=1.0):
    output = tf.zeros([input_signal.shape[1], input_signal.shape[2]], tf.float32)
    snr_linear = 10 ** (snr_dB / 10.0)
    for i in range(input_signal.shape[0]):
        input = input_signal[i]
        shape = tf.dtypes.cast(tf.size(input), tf.float32)
        avg_energy = tf.reduce_mean(tf.abs(input) * tf.abs(input))
        noise_variance = avg_energy / snr_linear
        noisenormal = tf.random.normal([tf.dtypes.cast(shape, tf.int32)])
        noise = tf.sqrt(noise_variance) * noisenormal
        output_signal = input + noise
        output = tf.concat([output, output_signal], 0)
    output = output[1:]
    output = tf.reshape(output, [output.shape[0], 1, output.shape[1]])
    return output
