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
RUN apt-get install -y libpq-dev

RUN pip3 install --upgrade pip

RUN cd /
RUN curl -L -O https://github.com/one-color-low/ReBone_v2/archive/master.zip
RUN unzip master.zip && rm master.zip
RUN mv ReBone_v2-master ReBone_v2 && cd ReBone_v2
RUN pip3 install -r requirements.txt

CMD ["bash"]

