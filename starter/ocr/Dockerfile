# Tesseract OCR with Vietnamese language data, used by `python -m src.run ocr`.
FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-vie tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /data
ENTRYPOINT ["tesseract"]
