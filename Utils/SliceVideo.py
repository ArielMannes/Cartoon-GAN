import argparse
import cv2
from skimage.measure import compare_ssim as ssim
from PIL import Image

path_out = ''
frame_count = 0


def extract_images(path_in):
    global path_out, frame_count
    vidcap = cv2.VideoCapture(path_in)
    success, last_image = vidcap.read()
    count = 0
    while success: # and frame_count < 100:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, count * 100)
        success, image = vidcap.read()
        if image is None:
            break
        last_image = compare_and_save(last_image, image)
        count += 1


def compare_and_save(imageA, imageB):
    global path_out, frame_count
    _ssim = ssim(imageA, imageB, multichannel=True)
    # _psnr = psnr(imageA, imageB)
    if _ssim < 0.35:
        _image = Image.fromarray(imageB[:, :, ::-1], 'RGB')
        _image = _image.resize((256, 256), Image.ANTIALIAS)
        _image.save('{}/frame{}.jpg'.format(path_out, frame_count))
        frame_count += 1
        return imageB
    return imageA


# def psnr(imageA, imageB):
#     mse = np.mean((imageA - imageB) ** 2)
#     if mse == 0:
#         return 100
#     return 20 * math.log10(255.0 / math.sqrt(mse))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pathIn', help='path to video',
                        default='C:/Adi/Videos/Spirited Away/Spirited Away.avi')
    parser.add_argument('--pathIOut', help='path to save the frames',
                        default='../Spirited_Away_frames')
    args = parser.parse_args()
    path_out = args.pathIOut
    extract_images(args.pathIn)
