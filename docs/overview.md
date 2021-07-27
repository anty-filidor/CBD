# CYBER BULLYING END-TO-END ML APP

This repository contains source code for simple end-to-end cyberbullying detector
application. Produced artifacts are deployed in the following way:

![documentation](docs/deployment_diagram.png "Schema of artifacts deployment")

Users are able to classify **polish** tweets with Python interpreter that has installed
the `cbd_client` package (available [on PyPi](https://pypi.org/project/cbd-client/)) or
any other http protocol tool like [Postman](https://www.postman.com/). When they send
request to flask application, the deployed ML model is inferred to classify text into
one of following classes (non-harmful, cyberbullying, hate_speech).

## TLDR

In order to see how the app works go to 'demo' folder and follow the manual.

## Contents of repository

Repository consists of following modules:

```
.
├── client
├── demo
├── docs
└── ml_pipeline
    ├── app
    └── svm
```

### Client

This module contains source code of the 'cbd-client' package.

### Demo

This module contains the Jupyter Notebook that shows how the solution works in reality.
Simple manual of environment configuration is also attached.

### ML Pipeline

This module contains source code of the web application and machine learning aspects of
the project. The 'app' directory stores the Flask application. The 'svm' directory
stores the entire machine learning 'pipeline' to (1) produce dataset, (2) train model,
(3) inference the model. It also includes data sets and evaluation scripts.

## Artifacts

Entire codebase is designed to produce three, following artifacts:

- **Machine Learning model** that classify input tweets as non-harmful, cyberbullying or
  hate_speech (can be found directly [in the repo](ml_pipeline/model.pkl))
- **Docker image** which contains the web application that serves the model inference
  (can be found [on Docker Hub](https://hub.docker.com/r/antyfilidor/cbd))
- **Python package** which facilitates clients to connect with application service and
  query the api to inference model (can be found
  [on PyPi](https://pypi.org/project/cbd-client/))

## Continuous integration

To facilitate the integration works we decided to implement two CI pipelines, so that
main artifacts can be automatically built. Due to the fact that GitHub was chosen as
DevOps platform, we implemented, so called, GitHub Actions. They are stored in the
'.github' folder.

`docker-image.yml` includes pipeline that automatically builds docker image that
contains the application to serve the CBD model. After docker image is built, the
pipeline publishes it on the Docker Hub.

`package-build.yml` includes pipeline that automatically builds and publish on pypi.org
the Python package that includes client code.

## Code quality

In order to create the code that can be relatively easy read by side programmers, we
decided to use [pre-commit](https://pre-commit.com/) tool that enables to run inspection
tools over the code at every attempt to commit changes in the repository. The
configuration file can be found [here](.pre-commit-config.yml). Briefly, it executes
lints like Black, Flake8 or MyPy and unit-test that have been created.

## Software engineering - discussion

It's obvious that this code can be implemented better, however the time limit to finish
the prototype enforced us to balance between transparency of solution and its
sophistication. Although some actions have been done to bring code of good quality, we
are aware that there are many things that can be done in the future in this field, like:

- move tests from local pre-commit hook to GitHub action,
- prepare MLOps pipeline that automatically process datasets, builds model and deploy
  it,
- write more tests (not only unit),
- modify rest api so that it can also handle batch requests,
- modify the `docker-image.yml` so that it can automatically update docker image on the
  GCP machine,
- in case of need to change model do deep one we can consider using GCP machine and
  Docker image that supports GPU acceleration,
- check all licenses of used libraries (use e.g. [FOSSA](https://fossa.com/) tool),
- prepare documentation of the code and publish it online (use e.g.
  [Sphinx](https://www.sphinx-doc.org/en/master/) and
  [Read the Docs](https://readthedocs.org/) server).
