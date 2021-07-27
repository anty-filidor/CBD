## Overview

This repository contains source code for simple end-to-end cyberbullying detector
application. Produced artifacts are deployed in the following way:

![documentation](docs/deployment_diagram.png "Schema of artifacts deployment")

Users are able to classify tweets with Python interpreter that has installed the
`cbd_client` package (available [on PyPi](https://pypi.org/project/cbd-client/)) or any
other http protocol tool like [Postman](https://www.postman.com/). When they send
request to flask application, the deployed ML model is inferred to classify text.

## TLDR

In order to see how the app works go to 'demo' folder and follow the manual.

## Contents

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
the project. The 'app' directory stores the Flask app. The 'svm' directory stores the
entire machine learning 'pipeline' to (1) produce dataset, (2) train model, (3)
inference the model. It also includes data sets and evaluation scripts.

## Continuous integration

## Code quality
