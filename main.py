import os

import cv2
from corner_detectors import harris_detector, fostner_detector
from util import create_color_gray, draw_points
import numpy as np

TYPE_LIST = ['harris', 'fostner']
PATH_TO_IMAGE = 'pictures/5d2369u-960.jpg'
PATH_TO_SAVE = 'example'
TYPE = TYPE_LIST[0]


def main() -> None:
    image = cv2.imread(PATH_TO_IMAGE, 0)
    points = []
    if TYPE == TYPE_LIST[0]:
        points = harris_detector(image, thr=0.03, const=0.04)
    elif TYPE == TYPE_LIST[1]:
        points = fostner_detector(image, thr=0.03, round_thr=0.04)
    color_gray_image = create_color_gray(image)
    draw_points(color_gray_image, points, color=[0, 0, 255])
    color_gray_image = color_gray_image.astype(np.uint8)
    cv2.imshow('dafaq', color_gray_image)
    cv2.waitKey(0)
    cv2.imwrite(os.path.join(PATH_TO_SAVE, TYPE + '_' + os.path.basename(PATH_TO_IMAGE)), color_gray_image)


if __name__ == '__main__':
    main()
