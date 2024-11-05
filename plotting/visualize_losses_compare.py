import numpy as np
import matplotlib.pyplot as plt
import os

filename1 = os.path.join('..','saved_models','model_1_2_2024_10_28','loss.npz')
filename2 = os.path.join('..','saved_models','model_1_2_2024_10_52','loss.npz')

npzfiles = np.load(filename1,allow_pickle=True)
training_loss_a = npzfiles['TRAINING_LOSS']
validation_loss_a = npzfiles['VALIDATION_LOSS']

npzfiles = np.load(filename2,allow_pickle=True)
training_loss_b = npzfiles['TRAINING_LOSS']
validation_loss_b = npzfiles['VALIDATION_LOSS']

plt.plot(training_loss_a, color='red', linestyle='solid', label='Training - 6 layers')
plt.plot(validation_loss_a, color='red', linestyle='dashed', label='Validation - 6 layers')
plt.plot(training_loss_b, color='blue', linestyle='solid', label='Training - 2 layers')
plt.plot(validation_loss_b, color='blue', linestyle='dashed', label='Validation - 2 layers')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
