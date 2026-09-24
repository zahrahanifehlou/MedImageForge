# Step 16: reproducible image for the MedImageForge CLI/API/UI.
#
# The dataset and artifacts are NEVER baked in (medical data + generated
# state stay out of the image — see .dockerignore). Mount them at runtime:
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" medimageforge info
#   docker run --rm -p 8000:8000 -v ... medimageforge serve
FROM python:3.10-slim

WORKDIR /app

# torch's CPU-only wheel lives on PyTorch's own index, not PyPI. We add it
# as an EXTRA index (not --index-url, which would replace PyPI): only the
# +cpu pin resolves there; every other package comes from PyPI. Using the
# pytorch index as the sole index makes modern pip discard PyPI-linked
# wheels with normalized-name mismatches and then fail to build sdists.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir -e .

# The image acts as the CLI itself: `docker run medimageforge qc` runs the
# qc command. Default subcommand is `info` — the always-safe smoke test.
ENTRYPOINT ["python", "-m", "medimageforge"]
CMD ["info"]
