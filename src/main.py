import csv
import cv2
import numpy as np
import os
from typing import Optional, Tuple


IMAGE_PARSE_COUNT = 3
CIRCLE_OUTLINE_COLOR = (255, 255, 0)


def write_csv(output_filename: str, circles: np.ndarray) -> None:
    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ['x', 'y', 'radius']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (x, y, r) in circles:
            writer.writerow({'x': x, 'y': y, 'radius': r})


def write_processed_frame(img: np.ndarray, circles: np.ndarray, output_filename: str) -> None:
    for (x, y, r) in circles:
        cv2.circle(img, (x, y), r, CIRCLE_OUTLINE_COLOR, 1)
    cv2.imwrite(output_filename, img)


def process_image(filename: str, input_dir: str = 'data/frames_raw', output_dir: str = 'data/frames_processed') -> None:
    img = cv2.imread(os.path.join(input_dir, filename))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)[:2]
    if circles is not None:
        circles = np.round(circles[0, :]).astype('int')
    write_processed_frame(img, circles, os.path.join(output_dir, filename))
    write_csv(os.path.join(output_dir, f'{filename.removesuffix(".png")}.csv'), circles)


def main() -> None:
    all_frames = os.listdir('data/frames_raw')
    for filename in all_frames[:IMAGE_PARSE_COUNT]:
        process_image(filename)



if __name__ == '__main__':
    main()