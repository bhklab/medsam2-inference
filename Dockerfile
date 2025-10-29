ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# Add uv binary from official release
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Use uv to install into system Python environment
ENV UV_SYSTEM_PYTHON=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . /app

# Install PyTorch with CUDA support
RUN uv pip install --system --no-cache \
    torch==2.5.1+cu124 \
    torchvision==0.20.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124


# Install MedSAM2 in editable mode with dev extras
RUN uv pip install --system --no-cache -e ./MedSAM2[dev]

# Install additional dependencies
RUN uv pip install --system --no-cache \
    "simpleitk>=2.5.2,<3" \
    "pydantic>=2.11.7,<3"

# Clean up all non-essential files
RUN rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Optional test
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

CMD ["/bin/bash"]
