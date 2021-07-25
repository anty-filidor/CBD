## Demo environment manual configuration

In order to run demo you need to have installed correct python environment, hence we
recommend following these steps:

### Create environment

We recommend to ues conda, however any 'venv' is sensible

```
conda create --name CBD_demo python=3.8
conda activate CBD_demo
```

### Install required packages:

```
pip install cbd-client jupyter
```

### Create and run Jupyter kernel from created env

```
python -m ipykernel install --user --name CBD_demo
jupyter notebook
```

### Run attached Jupyter notebook

1. Open Jupyter's web page
2. Navigate to './CBD/demo' directory
3. Open `demo.ipynb`
4. Before running the notebook - please select correct kernel (i.e. CBD_demo)
