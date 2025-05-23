FROM ubuntu:24.04

RUN apt-get update -qq \
    && apt-get install -y \
    python3-pip \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# we are inside a custom docker environment so we can happily overwrite
# with break-system-packages
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt --break-system-packages

# create workspace
RUN mkdir -p /ws
WORKDIR /ws

CMD [ "tail", "-f", "/dev/null" ]