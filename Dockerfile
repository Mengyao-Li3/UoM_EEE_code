FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
