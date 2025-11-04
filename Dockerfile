# For more information, please refer to https://aka.ms/vscode-docker-python
#FROM python:3.14
FROM ghcr.io/mlflow/mlflow:v3.6.0rc0

RUN mkdir /data/

# For building pyarrow for mlflow
RUN apt-get update && apt-get install -y git

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
RUN python -m pip install pylint
RUN python -m pip install optuna
RUN python -m pip install dvc 
RUN python -m pip install dvclive[image,plots,sklearn,markdown,torch]

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app && chown -R appuser /data
USER appuser

ENV PYTHONPATH=/workspaces/Wikilacra
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# MLflow Tracking Server
EXPOSE 5000

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD mlflow server --backend-store-uri sqlite:////db/mlflow.db --default-artifact-root /db/mlartifacts