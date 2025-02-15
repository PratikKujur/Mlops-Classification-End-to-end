FROM python:3.10

WORKDIR /app

# Copy all project files, including params.yaml
COPY . /app  

RUN pip install -r requirements.txt

CMD ["python3", "main.py"]
