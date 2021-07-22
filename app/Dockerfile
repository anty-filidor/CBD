FROM tensorflow/tensorflow:2.2.3-gpu-py3

# Set the working directory.
WORKDIR /usr/src/app

# Copy file with requirements
COPY requirements.txt .

# Copy all scripts required to run app
COPY classifier ./classifier
COPY main.py .

# Copy model
COPY dummy_model.joblib .

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
ENV MODEL_PATH=/usr/src/app/dummy_model.joblib
