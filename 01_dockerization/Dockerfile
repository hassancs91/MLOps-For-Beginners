# Multi-stage Dockerfile for Airline Sentiment Analysis API
# This Dockerfile creates an optimized production image for the FastAPI application

# Stage 1: Builder stage
# This stage is used to install dependencies and prepare the application
FROM python:3.11-slim as builder

# Set working directory for the builder stage
WORKDIR /build

# Copy only requirements first to leverage Docker cache layers
# This ensures that dependencies are only reinstalled if requirements.txt changes
COPY requirements.txt .

# Install dependencies in a virtual environment
# Using --no-cache-dir reduces image size by not caching pip packages
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# Stage 2: Production stage
# This stage creates the final, minimal image for running the application
FROM python:3.11-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk
# PYTHONUNBUFFERED: Ensures stdout and stderr are unbuffered for better logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Create a non-root user for security
# Running as non-root reduces the attack surface of the container
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Set the working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
# This includes all installed Python packages
COPY --from=builder /opt/venv /opt/venv

# Copy application code and model artifacts
# Using --chown ensures proper ownership for the non-root user
COPY --chown=appuser:appuser . .

# Copy the health check script
# This script will be used by Docker to monitor container health
COPY --chown=appuser:appuser healthcheck.py .

# Switch to non-root user for security
USER appuser

# Expose the port that FastAPI will run on
# This documents which port the application uses (doesn't actually publish it)
EXPOSE 8000

# Configure health check
# Docker will periodically run this command to check if the container is healthy
# --interval: How often to check health (every 30 seconds)
# --timeout: Maximum time to wait for health check to complete (10 seconds)
# --start-period: Grace period before health checks start (40 seconds for app startup)
# --retries: Number of consecutive failures before marking unhealthy (3 attempts)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python healthcheck.py || exit 1

# Set the default command to run the FastAPI application
# Using exec form ensures proper signal handling for graceful shutdown
# --host 0.0.0.0: Binds to all network interfaces (required for Docker)
# --port 8000: Port to listen on (matches EXPOSE directive)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]