import csv
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import pytesseract
import skimage.feature
import skimage.io
from typing import List, Optional, Tuple

logging.root.setLevel(logging.INFO)


IMAGE_PARSE_COUNT = 1
CIRCLE_OUTLINE_COLOR = (0.0, 1.0, 1.0)


Image = np.ndarray
Circle = Tuple[int, int, int]


def write_csv(output_filename: str, droplet1: Circle, droplet2: Circle, live_time: float) -> None:
    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = [
            'Droplet 1 X', 'Droplet 1 Y', 'Droplet 1 Radius',
            'Droplet 2 X', 'Droplet 2 Y', 'Droplet 2 Radius',
            'Live Time',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'Droplet 1 X': droplet1[0], 'Droplet 1 Y': droplet1[1], 'Droplet 1 Radius': droplet1[2],
            'Droplet 2 X': droplet2[0], 'Droplet 2 Y': droplet2[1], 'Droplet 2 Radius': droplet2[2],
            'Live Time': live_time,
        })


def write_processed_frame(img: Image, circles: List[Circle], output_filename: str) -> None:
    img_with_circles = draw_circles(img, circles)
    skimage.io.imsave(output_filename, (img_with_circles * 255).astype(np.uint8))


def find_circles(img: Image, possible_radii: np.ndarray = np.arange(500, 700, 50)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logging.info("Finding circles...")
    logging.info(f"Shapes {img.shape}, {possible_radii}")
    hough_res = skimage.transform.hough_circle(img, possible_radii)
    accums, cx, cy, radii = skimage.transform.hough_circle_peaks(
        hough_res, possible_radii, total_num_peaks=2, min_xdistance=10, min_ydistance=10)
    radii = radii.astype(np.int64)
    logging.info("Done finding circles.")
    return cx, cy, radii


def draw_circles(img: Image, circles: List[Circle], color: Tuple[float] = CIRCLE_OUTLINE_COLOR) -> Image:
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
    img = skimage.transform.resize(img, (img.shape[0] // scaler, img.shape[1] // scaler))
    return img


def find_live_time_in_image(img: Image) -> float:
    img = PIL.Image.fromarray(np.uint8(img) * 255)
    try:
        image_text = pytesseract.image_to_string(img)
    except pytesseract.TesseractNotFoundError as e:
        logging.error('Tesseract not found, returning 0 instead: %s', e)
        return 0
    logging.info('Found text: %s', image_text)
    lines = [v.split(':', 1) for v in image_text.split('\n')]
    live_time_dict = dict([(v[0], v[1].strip()) for v in lines if len(v) == 2])
    if not 'Live Time' in live_time_dict:
        logging.warning('Could not find Live Time in text. Keys: %s',
            live_time_dict.keys())
        return -1
    live_time = live_time_dict['Live Time']
    live_time_segments = live_time.split('.')[0].split(':')
    live_time_seconds = (
        int(live_time_segments[0]) * 3600 +
        int(live_time_segments[1]) * 60 +
        int(live_time_segments[2]))
    return live_time_seconds


def process_image(filename: str, input_dir: str = 'data/frames_raw', output_dir: str = 'data/frames_processed') -> None:
    img = skimage.io.imread(os.path.join(input_dir, filename))
    img = edges_image(img)
    img = prep(img)
    circle_proportion = 0.3
    cx, cy, radii = find_circles(
        img, possible_radii=np.arange(
            img.shape[1] * circle_proportion - 100,
            img.shape[1] * circle_proportion + 100,
            10))
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1) * np.ones([len(img), len(img[0]), 3])   
    circles = list(zip(cx, cy, radii))
    live_time = find_live_time_in_image(img)
    write_processed_frame(img, circles, os.path.join(output_dir, filename))
    write_csv(os.path.join(output_dir, f'{filename.removesuffix(".png")}.csv'), circles[0], circles[1], live_time)


def main() -> None:
    all_frames = os.listdir('data/frames_raw')
    for filename in all_frames[:IMAGE_PARSE_COUNT]:
        logging.info(f"Processing {filename}...")
        process_image(filename)
        logging.info(f"Done processing {filename}.")


if __name__ == '__main__':
    main()