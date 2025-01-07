# hash:sha256:507dd3f21133247c82efaf141b925feb6fd232a0a5788b05c5c47a9b2a3000da
FROM registry.codeocean.com/codeocean/miniconda3:4.8.2-python3.8-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        h5py==2.10.0 \
        matplotlib==3.3.1 \
        numpy==1.19.1 \
    && conda clean -ya
