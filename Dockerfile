# set base image (host OS)
FROM python:3.6

# set the working directory in the container
WORKDIR /chatbot-pkl-docker

# copy the dependencies file to the working directory
COPY requirements.txt requirements.txt
# install dependencies
RUN pip install -r requirements.txt

COPY . .