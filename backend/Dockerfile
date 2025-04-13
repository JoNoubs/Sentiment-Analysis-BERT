FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install 'accelerate>=0.26.0' tf-keras flask huggingface_hub[hf_xet]

COPY backend/ ./backend/
COPY data/ ./data/
COPY models/ ./models/

VOLUME /app/models

EXPOSE 5000

ENV PYTHONPATH=/app
CMD ["/bin/sh", "-c", "python -m backend.best_model && flask --app backend.api run --host=0.0.0.0 --port=5000"]