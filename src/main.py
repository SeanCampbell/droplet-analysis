import cv2
import numpy as np
import os


IMAGE_PARSE_COUNT = 3


def process_image(filename, input_dir='data/frames_raw', output_dir='data/frames_processed'):
    img = cv2.imread(os.path.join(input_dir, filename))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
    cv2.imwrite(os.path.join(output_dir, filename), img)


def main():
    all_frames = os.listdir('data/frames_raw')
    for filename in all_frames[:IMAGE_PARSE_COUNT]:
        process_image(filename)



if __name__ == '__main__':
    main()