FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --no-dev && rm -rf $POETRY_CACHE_DIR

# Copy application code
COPY . .

# Install the package
RUN poetry install --no-dev

# Create non-root user
RUN addgroup --gid 1001 --system appgroup && \
    adduser --no-create-home --shell /bin/false --disabled-password --uid 1001 --system --group appuser

# Change ownership of the app directory
RUN chown -R appuser:appgroup /app
USER appuser

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/evaluation_results

# Expose port for future web interface
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "from freerouter.config import get_config; get_config()" || exit 1

# Default command
CMD ["poetry", "run", "freerouter", "--help"]