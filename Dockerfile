FROM ubuntu:16.04

# virtualenv and git needed for run_tests.sh
# libsqlite3-dev needed for analysis-py-utils (for tests)
RUN apt-get -y update && apt-get -y upgrade && \
    apt-get -y install python python3 \
                       python-pip python3-pip \
                       virtualenv git \
                       libsqlite3-dev

# Install Python 3.6 and 3.7 to test against those versions.

# Dependencies needed to install and run 3.6 and 3.7
RUN apt-get -y install curl zlib1g-dev libssl-dev libffi-dev

RUN curl https://www.python.org/ftp/python/3.6.10/Python-3.6.10.tgz | tar xvz && \
    cd Python-3.6.10 && \
    ./configure && \
    make && \
    make altinstall

RUN curl https://www.python.org/ftp/python/3.7.6/Python-3.7.6.tgz | tar xvz && \
    cd Python-3.7.6 && \
    ./configure && \
    make && \
    make altinstall

CMD /bin/bash
