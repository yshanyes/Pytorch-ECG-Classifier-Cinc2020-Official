FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime
# FROM nvidia/cuda:10.1-cudnn7-devel

## FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime
## FROM pytorch/pytorch:nightly-runtime-cuda10.0-cudnn7
## FROM python:3.5.6-stretch

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER yangshanbuaa@163.com
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt-get etc.

## Do not edit if you have a requirements.txt
#RUN pip install --upgrade pip -i https://pypi.douban.com/simple/ --trusted-host pypi.douban.com
#RUN pip install --upgrade pip

RUN pip install -i https://pypi.douban.com/simple/ -r requirements.txt --trusted-host pypi.douban.com
#RUN pip install -r requirements.txt

