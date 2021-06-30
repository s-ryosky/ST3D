# From https://github.com/ufoym/deepo/blob/master/docker/Dockerfile.pytorch-py36-cu90

# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       1.8.0  (conda)
# ==================================================================

FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
ENV CONDA_DIR /opt/conda
ENV PATH ${CONDA_DIR}/bin:$PATH
RUN mkdir -p ${CONDA_DIR} && \
    echo export PATH=${CONDA_DIR}/bin:'$PATH' > /etc/profile.d/conda.sh
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        wget \
        git \
        vim \
        fish \
        libsparsehash-dev \
        zip \
        unzip \
        tar \
        sudo \
        curl \
        rsync \
        gnuplot-x11 \
        ghostscript \
        texlive-extra-utils \
        apt-utils \
        gcc \
        g++ \
        graphviz \
        pkg-config \
        zlib1g-dev \
        libprotobuf-dev \
        protobuf-compiler \
        software-properties-common \
        python-software-properties \
        python3-dev \
        python3-pip \
        python3-tk \
        python3-dbg \
        python3-wheel \
        python3-setuptools \
        libhdf5-dev \
        libopenblas-dev \
        liblapack-dev \
        libblas-dev \
        libgl1-mesa-dev \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        gfortran \
        locales \
        libboost-all-dev
# ==================================================================
# python
# ------------------------------------------------------------------
WORKDIR /opt
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh
ARG python_version=3.6.7
RUN conda install -y python=${python_version}
RUN pip install -U pip
RUN conda install \
        cudatoolkit==10.1.243 \
        h5py==2.10.0 \
        matplotlib==3.2.1 \
        mkl==2020.1 \
        nose==1.3.7 \
        notebook==6.1.1 \
        numpy==1.19.2 \
        pandas==1.1.0 \
        pillow==7.1.2 \
        pyyaml==5.3.1 \
        scikit-image==0.16.2 \
        scikit-learn==0.20.1 \
        scipy==1.4.1 \
        six==1.15.0
RUN pip install \
        addict==2.2.1 \
        cachetools==4.1.1 \
        cython==0.29.21 \
        descartes==1.1.0 \
        easydict==1.9 \
        filterpy==1.4.5 \
        fire==0.3.1 \
        future==0.18.2 \
        imagecorruptions==1.1.0 \
        joblib==0.16.0 \
        jupyter==1.0.0 \
        llvmlite==0.31.0 \
        motmetrics==1.1.3 \
        numba==0.47.0 \
        objgraph==3.4.1 \
        open3d-python==0.7.0.0 \
        opencv-python==4.3.0.38 \
        opencv-contrib-python==4.3.0.38 \
        optuna==2.6.0 \
        optuna-dashboard==0.3.1 \
        protobuf==3.12.4 \
        psutil==5.7.2 \
        pybind11==2.5.0 \
        pyquaternion==0.9.5 \
        pytest-runner==5.2 \
        seaborn==0.10.1 \
        shapely==1.7.0 \
        terminaltables==3.1.0 \
        tqdm==4.46.0 \
        vtk==9.0.1
# ==================================================================
# pytorch & tensorflow
# ------------------------------------------------------------------
RUN conda install \
        pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
RUN pip install \
        tensorflow==2.2.0 \
        lingvo==0.7.2 \
        tensorboardX==2.1
# ==================================================================
# dataset devkits
# ------------------------------------------------------------------
RUN pip install \
        pycocotools==2.0.1 \
        nuscenes-devkit==1.1.2
# ==================================================================
# Waymo Open Dataset devkit
# ------------------------------------------------------------------
ARG BAZEL_VERSION=3.1.0
RUN git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od && \
    cd waymo-od && \
    git checkout ec980535635e71645bc2ed437f22fda69d946bd0 && \
    wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    /bin/bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    ./configure.sh && \
    bazel clean && \
    bazel test waymo_open_dataset/metrics:all && \
    bazel build waymo_open_dataset/metrics/ops/... && \
    bazel build waymo_open_dataset/metrics/tools/... && \
    cp -r ./bazel-bin/waymo_open_dataset/* ./waymo_open_dataset/
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
# ==================================================================
# spconv
# use a commit 8da6f967fb9a054d8870c3515b1b44eca2103634 because Det3D only supports spconv1.0
# ------------------------------------------------------------------
RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.68.0/source/boost_1_68_0.tar.gz
RUN tar xzvf boost_1_68_0.tar.gz
RUN cp -r ./boost_1_68_0/boost /usr/include
RUN rm -rf ./boost_1_68_0
RUN rm -rf ./boost_1_68_0.tar.gz
RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4.tar.gz
RUN tar zxvf cmake-3.14.4.tar.gz
RUN cd ./cmake-3.14.4 && ./configure && make && make install && cd ..
RUN rm -rf ./cmake-3.14.4 && rm -rf ./cmake-3.14.4.tar.gz
RUN git clone --depth 99 --recursive https://github.com/traveller59/spconv.git
RUN cd ./spconv && \
    git checkout 8da6f967fb9a054d8870c3515b1b44eca2103634 && \
    python setup.py bdist_wheel && \
    pip install /opt/spconv/dist/spconv-1.0-cp36-cp36m-linux_x86_64.whl && \
    cd ..
# ==================================================================
# OpenPCDet
# ------------------------------------------------------------------
RUN git clone --depth 99 https://github.com/s-ryosky/ST3D
RUN cd ./ST3D && \
    python setup.py develop && \
    cd ..
# ==================================================================
# numba
# ------------------------------------------------------------------
ENV NUMBA_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
# ==================================================================
# others
# ------------------------------------------------------------------
# Set the locale
RUN locale-gen ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja:en
ENV LC_ALL ja_JP.UTF-8
# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

# User env
ARG USERNAME=docker
ARG UID=1000
ARG GID=1000
RUN useradd -m $USERNAME && \
    echo "$USERNAME:$USERNAME" | chpasswd && \
    usermod --shell /bin/bash $USERNAME && \
    usermod -aG sudo $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    usermod --uid $UID $USERNAME && \
    groupmod --gid $GID $USERNAME
