FROM nvidia/cuda:11.4.3-devel-ubuntu18.04
RUN apt-get update
RUN apt-get install -y python3.7 curl python3-distutils libboost-dev
RUN ln -s /usr/bin/python3.7 /usr/bin/python
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py
RUN mkdir /deepcube
COPY ./requirements.txt /deepcube
WORKDIR /deepcube
RUN pip install -r requirements.txt
ADD ./ /deepcube
WORKDIR /deepcube/cpp
RUN make
ENV PYTHONPATH /deepcube
ENV CUDA_VISIBLE_DEVICES 0

WORKDIR /deepcube/interface
CMD ["python", "server.py"]
