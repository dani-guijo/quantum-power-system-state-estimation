FROM python:3.9-slim-buster

COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "backend.py"]