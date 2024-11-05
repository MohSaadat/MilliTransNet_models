import numpy as np
import matplotlib.pyplot as plt
import os

filename = os.path.join('..','saved_models','model_30_1_2024_18_59','loss.npz')
npzfiles = np.load(filename,allow_pickle=True)
training_losses = npzfiles['TRAINING_LOSS']
validation_losses = npzfiles['VALIDATION_LOSS']

training_loss_d = []
training_loss_gan = []
for losses in training_losses:
    training_loss_d.append(losses[0])
    training_loss_gan.append(losses[1])

validation_loss_d = []
validation_loss_gan = []
for losses in validation_losses:
    validation_loss_d.append(losses[0])
    validation_loss_gan.append(losses[1])

plt.plot(training_loss_d, color='red', linestyle='solid', label='Training - discriminator')
plt.plot(validation_loss_d, color='blue', linestyle='dashed', label='Validation - discriminator')
plt.plot(training_loss_gan, color='red', linestyle='solid', label='Training - GAN')
plt.plot(validation_loss_gan, color='blue', linestyle='dashed', label='Validation - GAN')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
