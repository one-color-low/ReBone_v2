FROM tensorflow/tensorflow:1.8.0-gpu-py3

WORKDIR "/"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf-8

# install packages
## Preperation
RUN apt-get update -y --fix-missing
RUN apt-get install -y apt-utils
RUN apt-get install -y python3-pip

## for ffmpeg
RUN add-apt-repository ppa:jonathonf/ffmpeg-3
RUN apt update -y
RUN apt install -y ffmpeg libav-tools x264 x265

## for vmd-lifting
RUN apt-get install -y python3-pip
RUN apt-get install -y python3-tk
RUN apt-get install -y python3-pyqt5
RUN apt-get install -y cmake

## others
RUN apt-get install -y wget
RUN apt-get install -y vim

# install python packages
## Preparation
RUN pip3 install --upgrade pip

## for ffmpeg
RUN pip3 install ffmpeg-python

## for vmd-lifting
#RUN pip3 install tensorflow-gpu
#RUN pip3 install tensorflow
RUN pip3 install scikit-image
RUN pip3 install opencv-python==3.4.2.17

## for vc
RUN pip3 install librosa
RUN pip3 install pyworld
## ↑エラーが出るがなぜかsuccessfilly installed になる(要調査)

## for app
RUN pip3 install flask
RUN pip3 install flask_sqlalchemy
RUN pip3 install moviepy

# ---for deploy---
#RUN curl -L -O https://github.com/one-color-low/ReBone_v2/archive/fix-for-exact-deploy.zip
#RUN unzip master.zip && rm master.zip
#RUN unzip fix-for-exact-deploy.zip && rm fix-for-exact-deploy.zip

#RUN mv ReBone_v2-fix-for-exact-deploy ReBone_v2

## setups
#RUN /ReBone_v2/rebone_VC/setup.sh
#RUN unzip pretrain_data.zip && rm pretrain_data.zip
#RUN chmod +x /ReBone_v2/rebone_vmdl/setup.sh
#RUN /ReBone_v2/rebone_vmdl/setup.sh

# ---for local build---
RUN mkdir ReBone_v2
COPY ./ /ReBone_v2

## setups
RUN chmod +x /ReBone_v2/rebone_vmdl/setup.sh
RUN /ReBone_v2/rebone_vmdl/setup.sh
RUN mv /data /ReBone_v2/rebone_vmdl

CMD ["bash"]

