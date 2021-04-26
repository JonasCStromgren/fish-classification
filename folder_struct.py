import os
import shutil
import random
import glob

data_dir = 'Fiskar' # original image folder

# get the list of fish class folders
fish_dir = os.listdir(data_dir)

# create train, valid and test imgage folders and move randomly selected fish images into them
if os.path.isdir('train') is False:
    os.makedirs('train')
    os.makedirs('valid')
    os.makedirs('test')

    for fn in fish_dir:
        os.makedirs(f'train/{fn}')
        os.makedirs(f'valid/{fn}')
        os.makedirs(f'test/{fn}')

        for i in random.sample(os.listdir(f'{data_dir}/{fn}'), 500):
            shutil.move(f'{data_dir}/{fn}/{i}', f'train/{fn}')
        for i in random.sample(os.listdir(f'{data_dir}/{fn}'), 100):
            shutil.move(f'{data_dir}/{fn}/{i}', f'valid/{fn}')
        for i in random.sample(os.listdir(f'{data_dir}/{fn}'), 50):
            shutil.move(f'{data_dir}/{fn}/{i}', f'test/{fn}')
