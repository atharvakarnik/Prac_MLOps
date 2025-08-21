FROM python:3.11-slim

ARG VERSION
LABEL org.label-schema.version=$VERSION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY webapp/ /app/

EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["app.py"]