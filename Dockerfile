# InfraSight, CPU-only image for Hugging Face Spaces.
#
# The CPU build of torch ships without the bundled CUDA libraries, which is most of the
# weight of the default wheel. Measured cost on CPU is about 2 s per photo, so there is
# no reason to carry them.
FROM python:3.11-slim

# ultralytics pulls in opencv-python, the GUI build, which needs libgl1 and libglib at
# import time. Without them the container dies on `import cv2`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as uid 1000.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface \
    USE_TF=0

WORKDIR $HOME/app

# Install torch from the CPU index first so the dependency resolver never reaches for a
# CUDA wheel, then everything else.
#
# --no-compile skips the .pyc files. Python writes them on first import instead, which
# costs a little startup time once and saves several hundred MB in the image.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --no-compile --index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0 torchvision==0.21.0 \
    && pip install --no-cache-dir --no-compile -r requirements.txt

COPY --chown=user config/ ./config/
COPY --chown=user src/ ./src/
COPY --chown=user webapp/ ./webapp/

# SQLite history and annotated images. On a Space this is wiped by every restart, which
# is why app.ephemeral_storage exists.
RUN mkdir -p data/processed/analysis_results

EXPOSE 7860

# A cold start pays for the Depth Anything download and the weights fetch. Doing it here
# would bake ~150 MB into the image, so it stays a first-request cost instead.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health')"

CMD ["streamlit", "run", "webapp/app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.maxUploadSize=10", \
     "--browser.gatherUsageStats=false"]
