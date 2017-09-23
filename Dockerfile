FROM tensorflow/tensorflow:latest-py3
MAINTAINER knkski

RUN pip install keras

VOLUME /atai

EXPOSE 6006
EXPOSE 8888

WORKDIR /atai
