FROM python:3.10.13-slim-bookworm

COPY . /src

RUN pip install -r /src/requirements.txt