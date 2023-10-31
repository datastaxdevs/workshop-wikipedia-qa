FROM python:3.11.3-slim
ENV PYTHONUNBUFFERED 1

RUN pip install pip==23.2.1
RUN pip install pip-tools==7.1.0

# install cuda
#RUN apt-get -y update; apt-get -y install curl build-essential
#RUN curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
#RUN python install_gpu_driver.py

ADD requirements.txt .
RUN pip-sync requirements.txt 

ADD . .