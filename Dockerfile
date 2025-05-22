FROM ubuntu:24.04

RUN apt-get update -qq \
    && apt-get install -y \
    python3-pip \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app/

# we are inside a custom docker environment so we can happily overwrite
# with break-system-packages
RUN pip3 install -r requirements.txt --break-system-packages

CMD ["bash"]