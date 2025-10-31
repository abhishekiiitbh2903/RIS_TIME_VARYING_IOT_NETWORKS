from dataclasses import dataclass

@dataclass
class Config:
    neurons = 64
    inbits = 9
    cbits = 2 
    model_path="model.h5"
    log_path="log.csv"
    learning_rate = 0.001
    decay_steps=1000
    decay_rate=0.9
    staircase=True
    batch_size=1000
    epochs=1000
    result_path="result.csv"

