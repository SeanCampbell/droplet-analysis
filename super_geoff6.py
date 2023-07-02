import argparse
import glob
import multiprocessing
import os
import subprocess


DEFAULT_EXECUTABLE = './Geoff6.exe'
DEFAULT_BATCH_SIZE = 10
DEFAULT_FRAME_SKIP = 10


def parse_videos(filenames: list[str], executable: str, batch_size: int, frame_skip: int) -> None:
    with multiprocessing.Pool(batch_size) as pool:
        results = pool.map_async(
            subprocess.run, [[executable, f'"{f}"', str(frame_skip)] for f in filenames])
        pool.close()
        pool.join()
    results.get()


def parse_videos_in_directory(directory: str, executable: str, batch_size: int,
                              frame_skip: int) -> None:
    video_file_extensions = ['avi', 'mp4']
    filenames = []
    for extension in video_file_extensions:
        filenames += glob.glob(f'{directory}/*.{extension}')
    parse_videos(filenames, executable, batch_size, frame_skip)


def main():
    parser = argparse.ArgumentParser(
        prog='Super Geoff6', description='Runs geoff6 for multiple videos')
    parser.add_argument('directory')
    parser.add_argument('--executable', dest='executable', action='store',
                        required=False, default=DEFAULT_EXECUTABLE)
    parser.add_argument('--batch-size', dest='batch_size', action='store', type=int,
                        required=False, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--frame-skip', dest='frame_skip', action='store', type=int,
                        required=False, default=DEFAULT_FRAME_SKIP)
    args = parser.parse_args()
    parse_videos_in_directory(args.directory, args.executable, args.batch_size, args.frame_skip)


if __name__ == '__main__':
    main()