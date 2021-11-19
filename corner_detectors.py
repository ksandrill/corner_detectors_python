import numpy as np

from util import get_dX, get_dY


def harris_detector(grayscale_image: np.ndarray, thr: float, const: float, window_size: int = 3):
    normalized_image = grayscale_image / 255
    return get_interesting_points_by_harris_response(get_dX(normalized_image), get_dY(normalized_image), thr, const,
                                                     window_size)


def fostner_detector(grayscale_image: np.ndarray, thr: float, round_thr: float, window_size: int = 3):
    normalized_image = grayscale_image / 255
    return get_interesting_points_by_fostner_response(get_dX(normalized_image), get_dY(normalized_image), thr,
                                                      round_thr, window_size)


def get_interesting_points_by_harris_response(image_dX: np.ndarray, image_dY: np.ndarray, thr: float, const: float,
                                              window_size: int = 3) \
        -> list[(int, int)]:
    dots = []
    h, w = image_dX.shape
    square_dX = image_dX ** 2
    square_dY = image_dY ** 2
    dX_dY = image_dX * image_dY
    offset = window_size // 2
    for i in range(offset, h - offset):
        start_i = i - offset
        end_i = i + offset + 1
        for j in range(offset, w - offset):
            start_j = j - offset
            end_j = j + offset + 1
            ss_xx = np.sum(square_dX[start_i:end_i, start_j:end_j])
            ss_yy = np.sum(square_dY[start_i:end_i, start_j:end_j])
            ss_xy = np.sum(dX_dY[start_i:end_i, start_j:end_j])
            det = ss_xx * ss_yy - ss_xy ** 2
            trace = ss_xx + ss_yy
            response = det - const * trace ** 2
            if response >= thr:
                dots.append((j, i))  # x y
    return dots


def get_interesting_points_by_fostner_response(image_dX: np.ndarray, image_dY: np.ndarray, thr: float, round_thr: float,
                                               window_size: int = 3) \
        -> list[(int, int)]:
    dots = []
    h, w = image_dX.shape
    square_dX = image_dX ** 2
    square_dY = image_dY ** 2
    dX_dY = image_dX * image_dY
    offset = window_size // 2
    for i in range(offset, h - offset):
        start_i = i - offset
        end_i = i + offset + 1
        for j in range(offset, w - offset):
            start_j = j - offset
            end_j = j + offset + 1
            ss_xx = np.sum(square_dX[start_i:end_i, start_j:end_j])
            ss_yy = np.sum(square_dY[start_i:end_i, start_j:end_j])
            ss_xy = np.sum(dX_dY[start_i:end_i, start_j:end_j])
            det = ss_xx * ss_yy - ss_xy ** 2
            trace = ss_xx + ss_yy
            response = det / trace
            round_response = 4 * det / (trace ** 2)
            if response >= thr and round_response >= round_thr:
                dots.append((j, i))  # x y
    return dots
