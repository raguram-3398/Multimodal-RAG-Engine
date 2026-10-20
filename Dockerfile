# App image. CPU-only torch; BGE and the cross-encoder are downloaded at
# build time so the container never fetches models at startup.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# torch from the CPU index first: from PyPI, sentence-transformers would pull the
# CUDA build (several GB) on x86_64.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
COPY requirements-app.txt /tmp/
RUN pip install -r /tmp/requirements-app.txt

# HF Spaces runs the container as uid 1000; build as that user so the model
# cache and the app files are its own.
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

RUN python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1 PYTHONPATH=/home/user/app/src

COPY --chown=user src/multimodal_rag/ src/multimodal_rag/
COPY --chown=user data/chunks.jsonl data/
COPY --chown=user streamlit_app.py .

# No file watcher: nothing to hot-reload in a container, and walking every
# loaded module made transformers log ~2,400 lines of torchvision import errors.
EXPOSE 7860
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", \
     "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.fileWatcherType=none", "--client.toolbarMode=viewer"]
