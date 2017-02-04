FROM gcr.io/tensorflow/tensorflow:latest
MAINTAINER Hugo Lopes <hugo.l.lopes@google.com>

RUN pip install scikit-learn pyreadline Pillow
RUN sudo apt-get update
RUN pip install cython
RUN apt-get -y install wget tar ca-certificates
RUN apt-get install -y \
  autoconf \
  automake \
  autotools-dev \
  build-essential \
  checkinstall \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  libtool \
  python \
  python-imaging \
  python-tornado \
  wget \
  zlib1g-dev

RUN mkdir ~/temp \
  && cd ~/temp/ \
  && wget http://www.leptonica.com/source/leptonica-1.73.tar.gz \
  && tar -zxvf leptonica-1.73.tar.gz \
  && cd leptonica-1.73 \
  && ./configure \
  && make \
  && checkinstall \
  && ldconfig

RUN cd ~/temp/ \
  && wget https://github.com/tesseract-ocr/tesseract/archive/3.04.01.tar.gz \
  && tar -zxvf 3.04.01.tar.gz \
  && cd tesseract-3.04.01 \
  && ./autogen.sh \
  && mkdir ~/local \
  && ./configure --prefix=$HOME/local/ \
  && make \
  && make install

RUN sudo ldconfig
ENV PATH="/usr/local/lib:${PATH}"
RUN sudo apt-get install tesseract-ocr-eng
ENV PKG_CONFIG_PATH="/root/local/lib/pkgconfig"
RUN CPPFLAGS=-I/root/local/lib/pkgconfig pip install tesserocr
ENV LD_LIBRARY_PATH="/root/local/lib:${LD_LIBRARY_PATH}"
ENV TESSDATA_PREFIX="/usr/share/tesseract-ocr/tessdata"
RUN sudo ln -s  /root/local/lib/libtesseract.so.3 /usr/local/lib/libtesseract.so.3 #symlink

# keep original TF notebooks #RUN rm -rf /notebooks/*
#dont add #ADD *.ipynb /notebooks/
#WORKDIR /notebooks
#CMD ["/run_jupyter.sh"]
CMD ["/bin/bash"]