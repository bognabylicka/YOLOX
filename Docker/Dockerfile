ARG PYTORCH="1.8.1"
ARG CUDA="10.2"
ARG CUDNN="7"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y sudo
# Install YOLOX
RUN git clone https://github.com/bognabylicka/YOLOX.git /yolox
WORKDIR /yolox
RUN git checkout custom_coco
RUN pip3 install -U pip && pip3 install -r requirements.txt
RUN pip3 install -v -e .
# Install apex
RUN git clone https://github.com/NVIDIA/apex /apex
WORKDIR /apex
RUN pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# Install pycocotools
RUN pip3 install cython && pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
# Create a working directory
WORKDIR /yolox
