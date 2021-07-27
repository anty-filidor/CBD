FROM python:3.6-slim-buster

# Set the working directory.
WORKDIR /usr/src/app

# Copy file with requirements
COPY ml_pipeline/requirements.txt .

# Copy all scripts required to run ml_pipeline
COPY ml_pipeline/app ./app
COPY ml_pipeline/svm ./svm
COPY ml_pipeline/start_app.py ./main.py

# Copy model
COPY ml_pipeline/model.pkl .

# Install required linux packages
RUN apt update -y && apt install -y \
    apt-utils \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx

# Install required python packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Append path to python to PATH env variable
RUN export PATH="$PATH:/usr/local/bin/python"

# Create environment variable for better handling a model
ENV MODEL_PATH=/usr/src/app/model.pkl
