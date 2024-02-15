FROM python:3.9-slim

RUN apt-get update-y & \
    apt-get install -y -o Acquire::Retries=3 --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install \
    wheel \
    setuptools \