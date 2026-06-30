FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Falconsai model at build time
RUN python -c "from transformers import ViTForImageClassification; \
    ViTForImageClassification.from_pretrained('Falconsai/nsfw_image_detection', \
    num_labels=1, ignore_mismatched_sizes=True)"

COPY *.py .

RUN mkdir -p models uploads

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
