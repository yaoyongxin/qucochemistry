FROM python:3.7

RUN mkdir -p /opt/qucochemistry
WORKDIR /opt/qucochemistry

COPY ./ /opt/qucochemistry
RUN pip install --trusted-host pypi.python.org -r /opt/qucochemistry/deploy/requirements.txt
RUN pip install openfermionpyscf pyscf
RUN pip install -e .

EXPOSE 8888

ENTRYPOINT /bin/bash -c "jupyter notebook --NotebookApp.token='' --port=8888 --ip=0.0.0.0 --allow-root --no-browser ."
#CMD sleep 600
