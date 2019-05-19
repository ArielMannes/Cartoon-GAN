import glob

import cv2
import numpy as np
from tqdm import tqdm

path_to_dir = './'
spirited = 'Spirited_Away_frames'
siprited_smooth = 'Spirited_Away_frames_smooth'
flicker = 'flicker_images'

categories = ['animal', 'people', 'landscape', 'city', 'street', 'rural', 'train']


def make_np_arrays_flicker():
    global categories, path_to_dir, flicker
    for cat in categories:
        batch_glob = glob.glob('{}/{}/{}*.jpg'.format(path_to_dir, flicker, cat))
        print('\nloaded {}: containes {} pictures'.format(cat, len(batch_glob)))
        X = []
        for img in tqdm(batch_glob):
            im = cv2.imread(img)[:, :, ::-1]
            X.append(im)
        X = np.array(X)
        np.save('{}/Numpy_arrays/{}.npy'.format(path_to_dir, cat), X)
        print('\nsaved. shape:{}'.format(X.shape))


def train_test_split(train_percent=0.8):
    global path_to_dir, categories
    train = np.array([]).reshape((0, 256, 256, 3))
    test = np.array([]).reshape((0, 256, 256, 3))
    for cat in tqdm(categories):
        load_arr = np.load('{}/Numpy_arrays/{}.npy'.format(path_to_dir, cat))
        train_size = int(len(load_arr) * train_percent)
        train = np.vstack([train, load_arr[:train_size]])
        test = np.vstack([test, load_arr[train_size:]])
    np.random.shuffle(train)
    np.random.shuffle(test)
    np.save('{}/Numpy_arrays/flicker_train.npy'.format(path_to_dir), train)
    print('train: {}\nsave flicker_train.npy!'.format(train.shape))
    np.save('{}/Numpy_arrays/flicker_test.npy'.format(path_to_dir), test)
    print('test: {}\nsave flicker_test.npy!'.format(test.shape))


def make_np_array_spirited(_dir, file_name):
    global path_to_dir
    batch_glob = glob.glob('{}/{}/frame*.jpg'.format(path_to_dir, _dir))
    X = []
    for img in tqdm(batch_glob):
        im = cv2.imread(img)[:, :, ::-1]
        X.append(im)
    X = np.array(X)
    np.save('{}/Numpy_arrays/{}.npy'.format(path_to_dir, file_name), X)
    print('save {}.npy!'.format(file_name))


def main():
    global spirited, siprited_smooth
    make_np_arrays_flicker()
    train_test_split()
    make_np_array_spirited(spirited, 'spirited')
    make_np_array_spirited(siprited_smooth, 'siprited_smooth')
