## Flask API

The
[source](https://xaviervasques.medium.com/machine-learning-prediction-in-real-time-using-docker-and-python-rest-apis-with-flask-4235aa2395eb)
article and the [source code](https://github.com/xaviervasques/Online_Inference) that
was a skeleton of the final app.

### Building Docker image

```
cd app
docker build -t api -f Dockerfile .
docker run -it -p 5000:5000 api python3 main.py
```
