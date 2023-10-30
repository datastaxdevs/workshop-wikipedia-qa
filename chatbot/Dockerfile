FROM python:3.11.3-slim
ENV PYTHONUNBUFFERED 1

RUN apt-get -y update; apt-get -y install build-essential

RUN pip install pip==23.2.1
RUN pip install pip-tools==7.1.0

ADD requirements.txt .
RUN pip-sync requirements.txt 

# maybe makes it faster
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

ADD . .