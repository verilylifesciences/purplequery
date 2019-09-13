FROM ubuntu:16.04

# virtualenv and git needed for run_tests.sh
# libsqlite3-dev needed for analysis-py-utils (for tests)
RUN apt-get -y update && apt-get -y upgrade && \
    apt-get -y install python python3 \
                       python-pip python3-pip \
                       virtualenv git \
                       libsqlite3-dev

CMD /bin/bash
