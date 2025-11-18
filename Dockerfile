FROM rayproject/ray:latest-py311-gpu

USER root

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create writable directory
WORKDIR /app

# Copy project
COPY . .

# Create venv + install deps
RUN uv venv --python 3.11 --clear && \
    . .venv/bin/activate && \
    uv sync --extra cuda126

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

CMD ["ray", "start", "--head"]
