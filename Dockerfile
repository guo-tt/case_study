FROM jupyter/datascience-notebook:latest

USER root

RUN apt-get update && apt-get install -y build-essential libpq-dev gcc
ENV PIP_NO_CACHE_DIR=1

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Singapore

COPY requirements.txt /apps/requirements.txt

WORKDIR /apps

RUN pip install -r requirements.txt --force-reinstall

EXPOSE 7860