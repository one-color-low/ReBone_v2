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
RUN add-apt-repository ppa:jonathonf/ffmpeg-3
RUN apt update -y
RUN apt install -y ffmpeg libav-tools x264 x265

RUN pip3 install --upgrade pip
RUN pip3 install certifi==2019.9.11
RUN pip3 install chardet==3.0.4
RUN pip3 install Click==7.0
RUN pip3 install decorator==4.4.0
RUN pip3 install ffmpeg-python==0.1.18
RUN pip3 install Flask==1.1.1
RUN pip3 install Flask-SQLAlchemy==2.4.1
RUN pip3 install future==0.18.1
RUN pip3 install gunicorn==19.9.0
RUN pip3 install idna==2.8
RUN pip3 install imageio==2.6.1
RUN pip3 install imageio-ffmpeg==0.3.0
RUN pip3 install itsdangerous==1.1.0
RUN pip3 install Jinja2==2.10.3
RUN pip3 install MarkupSafe==1.1.1
RUN pip3 install moviepy==1.0.1
RUN pip3 install numpy==1.17.3
RUN pip3 install Pillow==6.2.1
RUN pip3 install proglog==0.1.9
RUN pip3 install psycopg2==2.8.4
RUN pip3 install pyee==6.0.0
RUN pip3 install python-ffmpeg==1.0.8
RUN pip3 install requests==2.22.0
RUN pip3 install SQLAlchemy==1.3.10
RUN pip3 install tqdm==4.36.1
RUN pip3 install urllib3==1.25.6
RUN pip3 install Werkzeug==0.16.0
RUN pip3 install tensorflow-gpu==1.8.0
RUN pip3 install progressbar2==3.37.1
RUN pip3 install librosa==0.6.0
#RUN pip3 install ffmpeg==1.4
RUN pip3 install pyworld==0.2.8
RUN pip3 install wget==3.2
RUN pip3 install opencv-python==3.4.2.17
RUN pip3 install matplotlib==3.0.1
RUN pip3 install PyQt5
RUN pip3 install scikit-image

RUN cd /
RUN curl -L -O https://github.com/one-color-low/ReBone_v2/archive/master.zip
RUN unzip master.zip && rm master.zip
RUN mv ReBone_v2-master ReBone_v2 && cd ReBone_v2
#RUN pip3 install -r /ReBone_v2/requirements.txt
RUN echo y | apt-get install vim


CMD ["bash"]

