from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D,Reshape,Concatenate
import tensorflow_compression as tfc
from attention import AttentionLayer
from utils import DRblock


class Decoder(Model):
    def __init__(self, inbits, neurons):
        super(Decoder, self).__init__()
        self.neurons = neurons
        self.inbits = inbits
        self.inilayer = Conv1D(neurons * inbits, 1, activation='relu')
        self.reshape = Reshape((inbits, neurons))
        self.DRblock1 = DRblock(neurons)
        self.attention_layer1 = AttentionLayer()
        self.igdn1= tfc.layers.GDN(inverse=True)
        self.DRblock2 = DRblock(neurons)
        self.attention_layer2=AttentionLayer()
        self.igdn2= tfc.layers.GDN(inverse=True)
        self.attention_layer3=AttentionLayer()
        self.igdn3= tfc.layers.GDN(inverse=True)
        self.cu = Conv1D(neurons, 1)
        self.conv = Conv1D(1, 3, padding='same', activation='sigmoid')
        self.concat = Concatenate()

    def call(self, input):
        output = self.inilayer(input)
        noise_output = self.reshape(output)
        output =self.DRblock1(noise_output)
        output=self.attention_layer1(output)
        output=self.igdn1(output)
        output =self.DRblock2(output)
        output=self.attention_layer2(output)
        output=self.igdn2(output)
        output =self.concat([noise_output, output])
        output =self.attention_layer3(output)
        output=self.igdn3(output)
        output =self.cu(output)
        output = noise_output - output
        output = self.conv(output)
        return output



