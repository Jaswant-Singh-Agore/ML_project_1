# ✅ Use stable and secure Python 3.11 base
FROM python:3.11-slim-bookworm
WORKDIR /app
COPY . /app

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "application:application"]
