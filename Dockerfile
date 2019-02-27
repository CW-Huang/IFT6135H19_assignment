FROM pytorch/pytorch

WORKDIR /project

RUN pip install ipdb && \
    pip install matplotlib && \
    pip install torchvision
