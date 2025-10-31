from encoder import Encoder
from decoder import Decoder
from utils import *
from tensorflow.keras.models import Model


class Autoencoder(Model):
    def __init__(self, inbits, cbits, neurons):
        super(Autoencoder, self).__init__()
        self.cbits = cbits
        self.inbits = inbits
        self.neurons = neurons
        self.enc = Encoder(inbits, cbits, neurons)
        self.dec = Decoder(inbits, neurons)

    def call(self, input):
        ouputenc = self.enc(input)
        ouputenc = awgn(ouputenc,10) # Apan rician ya awgn switch krlenge for experiments, Making it Dynamic
        ouputdec = self.dec(ouputenc)
        return ouputdec
    


