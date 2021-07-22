FROM tensorflow/tensorflow:2.2.3-gpu-py3

# Set the working directory.
WORKDIR /usr/src/app

# Copy file with requirements
COPY requirements.txt .

# Create the model dir
RUN mkdir my-model

# Copy all scripts required to run app
COPY inference_lda.joblib ./my-model/inference_lda.joblib
COPY inference_nn.joblib ./my-model/inference_nn.joblib
COPY rest_api.py .
COPY utils.py .

# Install required linux packages
RUN apt update -y && apt install -y \
    apt-utils \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    rsync

# Install python packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# append path to python to PATH env variable
RUN export PATH="$PATH:/usr/local/bin/python"

# Create environment variables for better handling a motel
ENV MODEL_DIR=./my-model
ENV MODEL_FILE_LDA=inference_lda.joblib
ENV MODEL_FILE_NN=inference_nn.joblib
