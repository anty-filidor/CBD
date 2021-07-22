# docker build -t api -f Dockerfile .
# docker run -it -p 5000:5000 api python3 main.py
# import requests
#
# res = requests.post("http://0.0.0.0:5000/classify_string/2", json={"text": "lalala"})
# if res.ok:
#     print(res.json())
