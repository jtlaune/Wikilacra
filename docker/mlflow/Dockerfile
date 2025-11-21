# For more information, please refer to https://aka.ms/vscode-docker-python
FROM ghcr.io/mlflow/mlflow:v3.6.0rc0

RUN mkdir /data/
RUN mkdir -p /workspaces/Wikilacra/

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

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /workspaces/Wikilacra && chown -R appuser /data
USER appuser

ENV PYTHONPATH=/workspaces/Wikilacra
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# MLflow Tracking Server
EXPOSE 5000

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD mlflow server \
  --backend-store-uri sqlite:////db/mlflow.db \
  --artifacts-destination /db/mlartifacts \
  --serve-artifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "mlflow:5000,localhost:*"