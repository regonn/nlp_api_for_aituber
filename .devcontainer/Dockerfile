FROM  tensorflow/tensorflow:1.12.3-py3

RUN apt update
RUN apt install -y git

ADD . /home/aituber
WORKDIR /home/aituber

RUN pip install flask keras
