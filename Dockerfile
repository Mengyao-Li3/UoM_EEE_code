FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt

RUN git clone https://github.com/Non-Invasive-Bioelectronics-Lab/Modified_Autoencoder4_challenge.git
#RUN git clone https://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@github.com/Non-Invasive-Bioelectronics-Lab/Modified_Autoencoder4_challenge.git
#RUN git clone https://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@github.com/Non-Invasive-Bioelectronics-Lab/Autoencoder.git
