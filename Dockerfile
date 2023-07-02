FROM python:3.11

WORKDIR /deps
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-437.0.1-linux-x86_64.tar.gz
RUN tar -xf google-cloud-cli-437.0.1-linux-x86_64.tar.gz
RUN ./google-cloud-sdk/install.sh
# RUN source google-cloud-sdk/path.bash.inc
# RUN export CLOUDSDK_PYTHON=/usr/local/bin/python

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libgl1-mesa-glx -y

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

CMD "jupyter notebook --allow-root --ip 0.0.0.0"