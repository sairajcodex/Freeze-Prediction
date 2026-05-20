FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y tesseract-ocr tesseract-ocr-eng libgl1 libglib2.0-0

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
