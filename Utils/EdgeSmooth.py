import argparse
import os

import cv2
import numpy as np
from glob import glob
from tqdm import tqdm


def make_edge_smooth(path_in, img_size):
    file_list = glob('{}/*.*'.format(path_in))
    save_dir = '{}_smooth/'.format(path_in)

    ks = 5
    hks = ks // 2
    kernel = np.ones((ks, ks), np.uint8)
    gauss = cv2.getGaussianKernel(ks, 0)
    gauss = gauss * gauss.transpose(1, 0)

    for f in tqdm(file_list):
        # read and resize image
        file_name = os.path.basename(f)
        bgr_img = cv2.resize(cv2.imread(f), (img_size, img_size))

        # (1) detect edge pixels using a standard Canny edge detector
        edges = cv2.Canny(cv2.resize(cv2.imread(f, 0), (img_size, img_size)), 100, 200)

        # (2) dilate the edge regions
        dilation = cv2.dilate(edges, kernel)

        # (3) apply a Gaussian smoothing in the dilated edge regions
        h, w = edges.shape
        gauss_img = np.copy(bgr_img)
        for i in range(hks, h - hks):
            for j in range(hks, w - hks):
                if dilation[i, j] != 0:  # gaussian blur to only edge
                    for k in range(3):
                        gauss_img[i, j, k] = np.sum(
                            np.multiply(bgr_img[i - hks:i + hks + 1, j - hks:j + hks + 1, k], gauss))

        cv2.imwrite(os.path.join(save_dir, file_name), gauss_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pathIn', help='path to directory of images to smooth', default='../Spirited_Away_frames')
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    args = parser.parse_args()
    make_edge_smooth(args.pathIn, args.img_size)
