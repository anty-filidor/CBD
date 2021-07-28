## Google cloud service

Link to the settings of VM:
https://console.cloud.google.com/compute/instances?project=cyberbullying-detector.

Update application on the GC:

```
sudo docker pull antyfilidor/cbd:latest
sudo docker run -it -p 5000:5000 antyfilidor/cbd python3 main.py
```

After executing this commands similar output should be displayed:

```
root@instance-1:~# sudo docker run -it -p 5000:5000 antyfilidor/cbd python3 main.py
[CBD APP] [13:58:54] [INFO] msg: "Logger initialised successfully!"
[CBD APP] [13:58:54] [INFO] msg: "Loading CBD model"
[CBD APP] [13:58:54] [INFO] msg: "Path to model is /usr/src/app/model.pkl, exists? 'True'."
[CBD APP] [13:58:54] [INFO] msg: "Creating Flask application"
 * Serving Flask app 'app.rest_api' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
[CBD APP] [13:58:54] [WARNING] msg: " * Running on all addresses.
   WARNING: This is a development server. Do not use it in a production deployment."
[CBD APP] [13:58:54] [INFO] msg: " * Running on http://172.17.0.2:5000/ (Press CTRL+C to quit)"
[CBD APP] [13:58:54] [INFO] msg: " * Restarting with stat"
[CBD APP] [13:58:56] [INFO] msg: "Logger initialised successfully!"
[CBD APP] [13:58:56] [INFO] msg: "Loading CBD model"
[CBD APP] [13:58:56] [INFO] msg: "Path to model is /usr/src/app/model.pkl, exists? 'True'."
[CBD APP] [13:58:56] [INFO] msg: "Creating Flask application"
[CBD APP] [13:58:56] [WARNING] msg: " * Debugger is active!"
[CBD APP] [13:58:56] [INFO] msg: " * Debugger PIN: 144-196-665"
```

## Validate results of model

`perl ml_pipeline/svm/evaluate2.pl ml_pipeline/svm/data/results.txt > ml_pipeline/svm/data/output.txt`

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
