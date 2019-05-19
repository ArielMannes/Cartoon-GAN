import glob

import cv2
import numpy as np
from tqdm import tqdm

path_to_dir = '..'
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


def train_test_split(train_percent=0.85):
    global path_to_dir, categories
    load_arr = np.load('{}/Numpy_arrays/{}.npy'.format(path_to_dir, categories[0]))
    train_size = int(len(load_arr) * train_percent)
    train = load_arr[:train_size]
    test = load_arr[train_size:]

    for cat in tqdm(categories[1:]):
        load_arr = np.load('{}/Numpy_arrays/{}.npy'.format(path_to_dir, cat))
        train_size = int(len(load_arr) * train_percent)
        train = np.insert(train, 0, load_arr[:train_size], axis=0)
        test = np.insert(test, 0, load_arr[train_size:], axis=0)
        del load_arr

    np.random.shuffle(train)
    np.random.shuffle(test)

    print('train: {}\ttest: {}'.format(train.shape, test.shape))

    np.save('{}/Numpy_arrays/flicker_train.npy'.format(path_to_dir), train)
    print('save flicker_train.npy!'.format(train.shape))
    np.save('{}/Numpy_arrays/flicker_test.npy'.format(path_to_dir), test)
    print('save flicker_test.npy!'.format(test.shape))


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


if __name__ == "__main__":
    main()
