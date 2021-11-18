FROM python:3.8.12

RUN apt-get update \
    && apt-get install -y nodejs \
    && apt-get install -y npm

#####################
# INSTALL DEPENDENCIES
#####################
RUN apt-get update \
  && apt-get install -y build-essential \
  && apt-get install -y libhdf5-dev \
  && apt-get install -y python-setuptools \
  && apt-get install -y python-dev


#####################
# CREATE APP DIRECTORIES
#####################
ENV HOME=/usr/app

# Create app directory
RUN mkdir -p $HOME
WORKDIR $HOME

RUN mkdir -p $HOME/data
RUN mkdir -p $HOME/dscript
RUN mkdir -p $HOME/server


#####################
# BUILD SERVER
#####################
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN pip3 install --upgrade cython

RUN pip3 install -r requirements.txt

COPY ./data $HOME/data
COPY ./dscript $HOME/dscript
COPY ./server $HOME/server


#####################
# BUILD FRONT END
#####################
WORKDIR $HOME/server/frontend
RUN npm install
RUN npm run build

WORKDIR $HOME/server
CMD [ "gunicorn", "-w", "1", "-b", "0.0.0.0:80", "server.wsgi" ]
