FROM tensorflow/tensorflow:1.8.0-gpu-py3

WORKDIR "/"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf-8

# install packages
RUN apt-get update -y --fix-missing
RUN apt-get install -y apt-utils
RUN apt-get install -y python3-pip
RUN apt-get install -y python3-tk
RUN apt-get install -y python3-pyqt5
RUN apt-get install -y cmake
RUN apt-get install -y curl
RUN apt-get install -y unzip

RUN pip3 install --upgrade pip
RUN pip3 install scikit-image
RUN pip3 install opencv-python==3.4.2.17
RUN pip3 install numpy==1.16.0
RUN pip3 install tensorflow-gpu==1.8
RUN pip3 install progressbar2==3.37.1
RUN pip3 install librosa==0.6.0
RUN pip3 install ffmpeg==1.4
RUN pip3 install pyworld==0.2.8
RUN pip3 install wget==3.2
RUN pip3 install tqdm==4.31.1

# ここらへん以下は変えていく

# install VMD-Lifting
RUN curl -L -O https://github.com/errno-mmd/VMD-Lifting/archive/master.zip
RUN unzip master.zip && rm master.zip
RUN mv VMD-Lifting-master vmdl
RUN cd /vmdl && sh setup.sh

# install VC
RUN cd /
RUN curl -L -O https://github.com/apiss2/rebone-VC/archive/master.zip
RUN unzip master.zip && rm master.zip
RUN mv GAN-Voice-Conversion-master VC
RUN cd /VC && sh setup.sh
RUN unzip pretrain_data.zip && rm pretrain_data.zip

CMD ["bash"]

