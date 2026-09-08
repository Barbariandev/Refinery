FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client ca-certificates build-essential libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.8.0 torchvision==0.23.0 \
    && pip install -r requirements.txt
COPY sn125/ /app/sn125/
RUN mkdir -p /app/sn125/rounds /app/sn125/audit /root/.sn125 /root/.cache/huggingface
STOPSIGNAL SIGINT
ENTRYPOINT ["python3", "-m", "sn125"]
CMD ["--help"]
