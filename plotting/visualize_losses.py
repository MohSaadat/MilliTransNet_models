import numpy as np
import matplotlib.pyplot as plt
import os

filename = os.path.join('..','saved_models','model_30_1_2024_18_59','loss.npz')
npzfiles = np.load(filename,allow_pickle=True)
training_loss = npzfiles['TRAINING_LOSS']
validation_loss = npzfiles['VALIDATION_LOSS']

plt.plot(training_loss, color='red', linestyle='solid', label='Training')
plt.plot(validation_loss, color='blue', linestyle='solid', label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
