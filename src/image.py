import csv
import dataclasses
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import pytesseract
from skimage.color import rgb2gray
import skimage.feature
import skimage.io
from typing import List, Optional, Tuple

logging.root.setLevel(logging.INFO)


IMAGE_PARSE_COUNT = 1
CIRCLE_OUTLINE_COLOR = (0.0, 1.0, 1.0)


Image = np.ndarray
Circle = Tuple[int, int, int]


@dataclasses.dataclass
class DropletData:
    droplet1_x: int
    droplet1_y: int
    droplet1_radius: int
    droplet2_x: int
    droplet2_y: int
    droplet2_radius: int
    live_time: float


def find_circles(img: Image, possible_radii: np.ndarray = np.arange(500, 700, 50)) -> List[Circle]:
    # img = _edges_image(img)
    img = _prep(img)
    logging.info("Finding circles...")
    logging.info(f"Shapes {img.shape}, {possible_radii}")
    hough_res = skimage.transform.hough_circle(img, possible_radii)
    accums, cx, cy, radii = skimage.transform.hough_circle_peaks(
        hough_res, possible_radii, total_num_peaks=2, min_xdistance=10, min_ydistance=10)
    radii = radii.astype(np.int64)
    logging.info("Done finding circles.")
    circles = list(zip(cx, cy, radii))
    return circles


def draw_circles(img: Image, circles: List[Circle], color: Tuple[float] = CIRCLE_OUTLINE_COLOR) -> Image:
    print(circles)
    for circle in circles:
        center_x, center_y, radius = circle
        circy, circx = skimage.draw.circle_perimeter(center_y, center_x, radius)
        img[circy, circx] = color
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
    if len(live_time_segments) == 1:
        return int(live_time_segments[0])
    live_time_seconds = (
        int(live_time_segments[0]) * 3600 +
        int(live_time_segments[1]) * 60 +
        int(live_time_segments[2]))
    return live_time_seconds


def process_image(filename: str, input_dir: str = 'data/frames_raw', output_dir: str = 'data/frames_processed') -> None:
    img = skimage.io.imread(os.path.join(input_dir, filename))
    img = rgb2gray(img[:, :, :3])
    circle_proportion = 0.3
    circles = find_circles(
        img, possible_radii=np.arange(
            img.shape[1] * circle_proportion - 100,
            img.shape[1] * circle_proportion + 100,
            10))
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1) * np.ones([len(img), len(img[0]), 3])   
    live_time = find_live_time_in_image(img)
    droplet_data = DropletData(
        droplet1_x=circles[0][0],
        droplet1_y=circles[0][1],
        droplet1_radius=circles[0][2],
        droplet2_x=circles[1][0],
        droplet2_y=circles[1][1],
        droplet2_radius=circles[1][2],
        live_time=live_time,
    )
    write_outputs(img, filename, output_dir, droplet_data)


def write_csv(output_filename: str, droplet_data: DropletData) -> None:
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
            'Droplet 1 X': droplet_data.droplet1_x,
            'Droplet 1 Y': droplet_data.droplet1_y,
            'Droplet 1 Radius': droplet_data.droplet1_radius,
            'Droplet 2 X': droplet_data.droplet2_x,
            'Droplet 2 Y': droplet_data.droplet2_y,
            'Droplet 2 Radius': droplet_data.droplet2_radius,
            'Live Time': droplet_data.live_time,
        })


def write_processed_frame(img: Image, output_filename: str, droplet_data: DropletData) -> None:
    img_with_circles = draw_circles(
        img, 
        [(droplet_data.droplet1_x, droplet_data.droplet1_y, droplet_data.droplet1_radius),
        (droplet_data.droplet2_x, droplet_data.droplet2_y, droplet_data.droplet2_radius)],
    )
    skimage.io.imsave(output_filename, (img_with_circles * 255).astype(np.uint8))


def write_outputs(img: Image, filename: str, output_dir: str, droplet_data: DropletData) -> None:
    write_processed_frame(img, os.path.join(output_dir, filename), droplet_data)
    write_csv(os.path.join(output_dir, f'{filename.removesuffix(".png")}.csv'), droplet_data)


def process_images_in_directory(directory: str) -> None:
    all_images = os.listdir('data/frames_raw')
    for filename in all_images[:IMAGE_PARSE_COUNT]:
        logging.info(f"Processing {filename}...")
        process_image(filename)
        logging.info(f"Done processing {filename}.")


# def _edges_image(img: Image, sigma: int = 25, low_threshold: float = 0.9, high_threshold: float = 0.95) -> Image:
#     gray_img = img
#     if gray_img.shape[-1] in (3, 4):
#         gray_img = gray_img[:,:,0]
#     # gray_img = skimage.feature.canny(
#     #     gray_img,
#     #     sigma=sigma,
#     #     low_threshold=low_threshold,
#     #     high_threshold=high_threshold,
#     #     use_quantiles=True)
#     return gray_img
    
    
def _prep(img: Image) -> Image:
    scaler = 4
    img = skimage.transform.resize(img, (img.shape[0] // scaler, img.shape[1] // scaler))
    return img