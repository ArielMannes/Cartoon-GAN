import argparse

import cv2


def extract_images(path_in, path_out):
    vidcap = cv2.VideoCapture(path_in)
    count, success = 0, True
    while success and count < 10:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
        success, image = vidcap.read()
        cv2.imwrite(path_out + '/frame{}.jpg'.format(count), image)
        count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pathIn', help='path to video',
                        default='C:/Adi/Videos/Spirited Away/Spirited Away.avi')
    parser.add_argument('--pathIOut', help='path to save the frames',
                        default='../Spirited_Away_frames')
    args = parser.parse_args()
    extract_images(args.pathIn, args.pathIOut)
