FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y wget && \
    apt-get install -y python3 python3-pip && \
    apt-get install -y tesseract-ocr libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev \
    libtbbmalloc2 libtbb-dev libopenexr-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgl1-mesa-dev libglib2.0-0

# Set the working directory in the container
WORKDIR /app

# Copy your Python script into the container at /app
COPY . /app/

# Install & download neccessary files required
RUN ./install.sh

# Install python requirements
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "main.py"]