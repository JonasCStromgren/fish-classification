import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


data_dir = 'Fiskar' # image folder

# get the list of fish species folders
fish_dir = [fn for fn in os.listdir(f'{data_dir}')]


# plotting 3 x 3 image matrix
fig = plt.figure(figsize = (8,6))
for i in range(9):
    fp = random.choice(os.listdir(f'{data_dir}/{fish_dir[i]}'))
    label = fish_dir[i]
    ax = fig.add_subplot(3, 3, i+1)

    # to plot without rescaling, remove target_size
    fn = image.load_img(f'{data_dir}/{fish_dir[i]}/{fp}', target_size = (100,100))
    plt.imshow(fn)
    plt.title(label)
    plt.axis('off')
plt.show()
