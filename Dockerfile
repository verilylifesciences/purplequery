FROM ubuntu:16.04

# Update apt and the system
RUN apt-get -y update && apt-get -y upgrade

# libsqlite3-dev needed for analysis-py-utils
# virtualenv needed for run_tests.sh
RUN apt-get -y install python python3 \
                       python-pip python3-pip \
		       virtualenv \
                       libsqlite3-dev

CMD /bin/bash
