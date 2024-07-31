FROM python:3.12

RUN apt-get update
RUN apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx

RUN pip install \
    matplotlib \
    scikit-image
    # opencv-python

WORKDIR /app