FROM jupyter/scipy-notebook

RUN mkdir my-model

COPY inference_lda.joblib ./my-model/inference_lda.joblib
COPY inference_nn.joblib ./my-model/inference_nn.joblib

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY test.json ./test.json

COPY api.py ./api.py

ENV MODEL_DIR=./my-model
ENV MODEL_FILE_LDA=inference_lda.joblib
ENV MODEL_FILE_NN=inference_nn.joblib

RUN python3 api.py
