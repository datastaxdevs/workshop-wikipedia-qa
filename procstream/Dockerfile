FROM python:3.11.3-slim
ENV PYTHONUNBUFFERED 1

RUN pip install pip==23.2.1
RUN pip install pip-tools==7.1.0

ADD requirements.txt .
RUN pip-sync requirements.txt 

ADD . .