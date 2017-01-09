FROM gcr.io/tensorflow/tensorflow:latest
MAINTAINER Hugo Lopes <hugo.l.lopes@google.com>

# Pillow needs libjpeg by default as of 3.0.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg8-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install scikit-learn pyreadline Pillow
# keep original TF notebooks #RUN rm -rf /notebooks/*
#dont add #ADD *.ipynb /notebooks/
#WORKDIR /notebooks
#CMD ["/run_jupyter.sh"]
CMD ["/bin/bash"]