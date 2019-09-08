FROM python:3.7-slim

RUN apt update
RUN apt install -y build-essential

RUN mkdir -p /opt/qucochemistry
WORKDIR /opt/qucochemistry

COPY ./ /opt/qucochemistry
RUN pip install --trusted-host pypi.python.org -r /opt/qucochemistry/deploy/requirements.txt

# required for running the tutorial notebooks
RUN pip install openfermionpyscf pyscf noisyopt
RUN pip install -e .

RUN apt purge -y build-essential

EXPOSE 8888

ENTRYPOINT /bin/bash -c "jupyter notebook --NotebookApp.token='' --port=8888 --ip=0.0.0.0 --allow-root --no-browser ."
