FROM tiangolo/uwsgi-nginx-flask:python3.6
LABEL maintainer="maintainer"

COPY requirements.txt /tmp/
RUN pip install -U pip && pip install -r /tmp/requirements.txt

COPY app /app

ENV NGINX_WORKER_PROCESSES auto