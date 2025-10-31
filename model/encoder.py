from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D,Reshape,Attention
import tensorflow_compression as tfc
from attention import AttentionLayer


class Encoder(Model):
    def __init__(self, inbits, cbits, neurons):
        super(Encoder, self).__init__()
        self.cbits = cbits
        self.inbits = inbits
        self.conv1 = Conv1D(neurons, 3, padding='same', activation='relu')
        self.attention_layer1 = AttentionLayer()
        self.gdn1= tfc.layers.GDN()
        self.conv2 = Conv1D(neurons, 3, padding='same', activation='relu')
        self.attention_layer2 = AttentionLayer()
        self.gdn2= tfc.layers.GDN()
        self.conv3 = Conv1D(neurons, 3, padding='same', activation='relu')
        self.attention_layer3 = AttentionLayer()
        self.gdn3= tfc.layers.GDN()
        self.reshape = Reshape((1, neurons * inbits))
        self.conv4 = Conv1D(cbits, 1)
        self.self_attention=Attention()
    def call(self, input):
        input_x=self.self_attention([input,input])
        output = self.conv1(input_x)
        output = self.attention_layer1(output)
        output=self.gdn1(output)
        output = self.conv2(output)
        output = self.attention_layer2(output)
        output=self.gdn2(output)
        output = self.conv3(output)
        output = self.attention_layer3(output)
        output=self.gdn3(output)
        output = self.reshape(output)
        output = self.conv4(output)
        return output


