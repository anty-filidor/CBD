

docker build -t api -f Dockerfile .
docker run -it -p 5000:5000 api python3 api.py

import requests
res = requests.post("http://0.0.0.0:5000/add_message/2", json={"my_text": "lalala"})
if res.ok:
    print(res.json())