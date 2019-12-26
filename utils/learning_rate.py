import numpy as np


#还没用

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.1
    epochs_drop = 100.0
    lrate = initial_lrate * np.power(drop,
           np.floor((1+epoch)/epochs_drop))
    return lrate

#已用

def poly_decay(lr=3e-4, max_epochs=100):
    def decay(epoch):
        # initialize the maximum number of epochs, base learning rate,
        # and power of the polynomial
        lrate = lr*(1-np.power(epoch/max_epochs, 0.9))
        return lrate
    return decay
