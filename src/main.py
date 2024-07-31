import csv
# import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.feature
import skimage.io
from typing import List, Optional, Tuple


logging.root.setLevel(logging.INFO)


IMAGE_PARSE_COUNT = 3
CIRCLE_OUTLINE_COLOR = (255, 255, 0)


Image = np.ndarray
Circle = Tuple[int, int, int]


def write_csv(output_filename: str, circles: List[Circle]) -> None:
    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ['x', 'y', 'radius']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (x, y, r) in circles:
            writer.writerow({'x': x, 'y': y, 'radius': r})


def write_processed_frame(img: Image, circles: List[Circle], output_filename: str) -> None:
    img_with_circles = draw_circles(img, circles)
    skimage.io.imsave(output_filename, img_with_circles)


def find_circles(img: Image, possible_radii: np.ndarray = np.arange(500, 700, 50)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logging.info("Finding circles...")
    logging.info(f"Shapes {img.shape}, {possible_radii}")
    hough_res = skimage.transform.hough_circle(img, possible_radii)
    accums, cx, cy, radii = skimage.transform.hough_circle_peaks(
        hough_res, possible_radii, total_num_peaks=2, min_xdistance=10, min_ydistance=10)
    radii = radii.astype(np.int64)
    logging.info("Done finding circles.")
    return cx, cy, radii


def draw_circles(img: Image, circles: List[Circle], color: Tuple[float] = (0.9, 0.08, 0.08)) -> Image:
    for center_x, center_y, radius in circles:
        circy, circx = skimage.draw.circle_perimeter(center_y, center_x, radius)
        img[circy, circx] = color
    return img


def edges_image(img: Image, sigma: int = 25, low_threshold: float = 0.9, high_threshold: float = 0.95) -> Image:
    gray_img = img
    if gray_img.shape[-1] in (3, 4):
        gray_img = gray_img[:,:,0]
    # gray_img = skimage.feature.canny(
    #     gray_img,
    #     sigma=sigma,
    #     low_threshold=low_threshold,
    #     high_threshold=high_threshold,
    #     use_quantiles=True)
    return gray_img


def prep(img: Image) -> Image:
    scaler = 4
    img = np.resize(img, (img.shape[0] // scaler, img.shape[1] // scaler))
    return img


def process_image(filename: str, input_dir: str = 'data/frames_raw', output_dir: str = 'data/frames_processed') -> None:
    img = skimage.io.imread(os.path.join(input_dir, filename))
    # img = cv2.imread(os.path.join(input_dir, filename))
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = edges_image(img)
    img = prep(img)
    # plt.imshow(img)
    skimage.io.imshow(img)
    circle_proportion = 0.3
    cx, cy, radii = find_circles(
        img, possible_radii=np.arange(
            img.shape[1] * circle_proportion - 100,
            img.shape[1] * circle_proportion + 100,
            10))
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1) * np.ones([len(img), len(img[0]), 3])   
    circles = list(zip(cx, cy, radii))
    draw_circles(img, circles)
    write_processed_frame(img, circles, os.path.join(output_dir, filename))
    write_csv(os.path.join(output_dir, f'{filename.removesuffix(".png")}.csv'), circles)


def main() -> None:
    all_frames = os.listdir('data/frames_raw')
    for filename in all_frames[:IMAGE_PARSE_COUNT]:
        logging.info(f"Processing {filename}...")
        process_image(filename)
        logging.info(f"Done processing {filename}.")



if __name__ == '__main__':
    main()