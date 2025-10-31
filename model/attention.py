import tensorflow as tf 
from tensorflow.keras.layers import Conv1D,Conv2D 
from tensorflow.keras.layers import Multiply, Permute, Add

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.num_features = input_shape[-2]
        self.num_channels = input_shape[-1]

        self.spatial_conv = Conv1D(self.num_channels, 1, activation='sigmoid')
        self.channel_conv = Conv1D(self.num_features, 1, activation='sigmoid')
        self.joint_channel_conv = Conv2D(1, (1, 1), activation='sigmoid')

    def call(self, inputs):
        input = inputs

        spatial_attention = self.spatial_conv(inputs)
        spatial_out = Multiply()([inputs, spatial_attention])

        x_permuted = Permute((2, 1))(inputs)
        channel_attention = self.channel_conv(x_permuted)
        channel_out = Permute((2, 1))(Multiply()([x_permuted, channel_attention]))

        x_expanded = tf.expand_dims(inputs, axis=-1)
        joint_channel_attention = self.joint_channel_conv(x_expanded)
        joint_channel_out = tf.squeeze(joint_channel_attention, axis=-1)
        joint_channel_out = Multiply()([inputs, joint_channel_out])

        out = Add()([spatial_out, channel_out, joint_channel_out])

        return out