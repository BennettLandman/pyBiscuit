# Multi-architecture container image for pyBiscuit
FROM python:3.11-slim

LABEL org.opencontainers.image.title="pyBiscuit" \
      org.opencontainers.image.description="Python port of BISCUIT: cortical surface registration, parcellation, curves, and shape metrics" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.authors="Bennett Landman <bennett.landman@vanderbilt.edu>" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYBISCUIT_ASSETS=/opt/biscuit/assets

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .[parc,viz] && \
    mkdir -p /opt/biscuit && \
    ln -s /usr/local/bin/run_biscuit /opt/biscuit/run_biscuit

ENTRYPOINT ["pybiscuit"]
CMD ["--help"]
