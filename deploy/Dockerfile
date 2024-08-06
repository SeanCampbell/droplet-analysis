FROM python:3.12

RUN apt-get update
RUN apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    tesseract-ocr

COPY ./requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app