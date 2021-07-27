## Flask API

The
[source](https://xaviervasques.medium.com/machine-learning-prediction-in-real-time-using-docker-and-python-rest-apis-with-flask-4235aa2395eb)
article and the [source code](https://github.com/xaviervasques/Online_Inference) that
was a skeleton of the final app.

## ML Part

To train final model I used source code of the winner of the PolEval 2019 competition -
Maciej Biesek. His source code can be found
[here](https://github.com/maciejbiesek/poleval-cyberbullying),

`perl ml_pipeline/svm/evaluate2.pl ml_pipeline/svm/data/results.txt > ml_pipeline/svm/data/output.txt`

### Building Docker image

```
cd app
docker build -t api -f Dockerfile .
docker run -it -p 5000:5000 api python3 main.py
```

## Setting up the environment

Prerequisites: Unix OS, conda

1. Navigate to project root
2. Create conda environment:

```
conda create --name CBD python=3.6.9
conda activate CBD
```

3. Install required packages:

```
pip install -r requirements.dev.txt
```

4. Create Jupyter kernel from created env

```
python -m ipykernel install --user --name CBD
```

## Client

### Building the package
