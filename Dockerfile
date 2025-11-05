#getting OS and Python image from DockerHub
FROM python:3.11.9-slim-bullseye

WORKDIR /docker

COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#copy all files (including large files)
COPY ./ ./

CMD ["python3", "-m", "flask", "--app", "predict_combine_models", "run", "--host=0.0.0.0"]
