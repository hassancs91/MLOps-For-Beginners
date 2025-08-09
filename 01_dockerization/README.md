# Phase 1: Dockerization

## Overview
This phase containerizes the airline sentiment analysis API, making it portable and ready for deployment. We've created a production-ready Docker image with health checks, security best practices, and optimized build layers.

## What's New in This Phase

### Files Added
- **`Dockerfile`**: Multi-stage build configuration for creating an optimized Docker image
- **`healthcheck.py`**: Python script for container health monitoring
- **`.dockerignore`**: Excludes unnecessary files from the Docker build context

### Key Improvements
- ✅ Multi-stage Docker build for smaller image size
- ✅ Non-root user for enhanced security
- ✅ Health check endpoint monitoring
- ✅ Optimized layer caching
- ✅ Python 3.11 slim base image

## Building the Docker Image

### Prerequisites
- Docker installed on your system
- All model files present in the `models/` directory

### Build Commands

1. **Navigate to the dockerization directory:**
```bash
cd 01_dockerization
```

2. **Build the Docker image:**
```bash
docker build -t airline-sentiment:v1 .
```

Expected output:
```
[+] Building...
=> [internal] load build definition from Dockerfile
=> [internal] load .dockerignore
=> [builder 1/3] FROM python:3.11-slim
=> [builder 2/3] COPY requirements.txt .
=> [builder 3/3] RUN python -m venv /opt/venv && ...
=> [stage-1 1/4] COPY --from=builder /opt/venv /opt/venv
=> exporting to image
=> naming to docker.io/library/airline-sentiment:v1
```

3. **Verify the image was created:**
```bash
docker images | grep airline-sentiment
```

## Running the Container

### Start the Container
```bash
docker run -d \
  --name sentiment-api \
  -p 8000:8000 \
  airline-sentiment:v1
```

### Verify Container is Running
```bash
docker ps
```

You should see:
```
CONTAINER ID   IMAGE                   STATUS                    PORTS
<id>           airline-sentiment:v1    Up X seconds (healthy)    0.0.0.0:8000->8000/tcp
```

## Testing the API

### 1. Check Health Endpoint
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

### 2. Test Model Information
```bash
curl http://localhost:8000/model-info
```

Expected response:
```json
{
  "model_version": "1.0.0",
  "model_type": "LogisticRegression",
  "accuracy": 0.8234,
  "training_samples": 11541,
  "features": 5000,
  "vectorizer_type": "TfidfVectorizer"
}
```

### 3. Make a Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Amazing flight! The crew was fantastic and very helpful."}'
```

Expected response:
```json
{
  "text": "Amazing flight! The crew was fantastic and very helpful.",
  "sentiment": "positive",
  "confidence": 95.3,
  "model_version": "1.0.0"
}
```

### 4. Access Interactive API Documentation
Open your browser and navigate to:
```
http://localhost:8000/docs
```

## Verifying Health Checks

### Check Docker Health Status
```bash
docker inspect sentiment-api --format='{{.State.Health.Status}}'
```

Should return: `healthy`

### View Health Check Logs
```bash
docker inspect sentiment-api --format='{{range .State.Health.Log}}{{.Output}}{{end}}'
```

### Test Health Check Script Directly
```bash
docker exec sentiment-api python healthcheck.py
echo "Exit code: $?"
```

Exit code 0 = healthy, 1 = unhealthy

## Container Logs

### View Application Logs
```bash
docker logs sentiment-api
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup
INFO:     Loading model artifacts...
INFO:     Model loaded successfully! Version: 1.0.0
INFO:     Application startup complete
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Follow Logs in Real-time
```bash
docker logs -f sentiment-api
```

## Stopping and Cleaning Up

### Stop the Container
```bash
docker stop sentiment-api
```

### Remove the Container
```bash
docker rm sentiment-api
```

### Remove the Image (if needed)
```bash
docker rmi airline-sentiment:v1
```

## Troubleshooting

### Container Won't Start
1. Check logs: `docker logs sentiment-api`
2. Verify model files exist in `models/` directory
3. Ensure port 8000 is not already in use: `lsof -i :8000`

### Health Check Failing
1. Check if the app is running: `docker exec sentiment-api ps aux`
2. Test health endpoint manually: `docker exec sentiment-api python healthcheck.py`
3. Review health check logs: `docker inspect sentiment-api --format='{{json .State.Health}}'`

### API Not Responding
1. Verify port mapping: `docker port sentiment-api`
2. Check firewall settings
3. Test from inside container: `docker exec sentiment-api curl localhost:8000/health`

### Model Not Loading
1. Verify model files are in the image: `docker exec sentiment-api ls -la models/`
2. Check file permissions: `docker exec sentiment-api ls -la models/`
3. Review startup logs for errors: `docker logs sentiment-api | grep ERROR`

### Cannot Remove Docker Image
If you see this error when trying to remove the image:
```
Error response from daemon: conflict: unable to delete airline-sentiment:v1 (must be forced) - container <id> is using its referenced image
```

This is **normal Docker behavior** - Docker protects you from deleting images that containers are using. Solutions:

1. **Remove the container first, then the image:**
```bash
docker rm sentiment-api     # or use container ID
docker rmi airline-sentiment:v1
```

2. **Force remove the image (use with caution):**
```bash
docker rmi -f airline-sentiment:v1
```

3. **Clean up all stopped containers and then remove image:**
```bash
docker container prune      # removes all stopped containers
docker rmi airline-sentiment:v1
```

## Success Criteria Checklist

- [ ] Docker image builds without errors
- [ ] Container starts and shows "healthy" status
- [ ] Health endpoint returns `{"status": "healthy", "model_loaded": true}`
- [ ] Model info endpoint shows correct model metadata
- [ ] Prediction endpoint returns sentiment with confidence score
- [ ] Container logs show successful model loading
- [ ] Health checks pass consistently (check 3-4 times)
- [ ] API documentation accessible at `/docs`

## Next Steps
Once all success criteria are met, you're ready to move to the next phase of the MLOps pipeline. The containerized application can now be:
- Deployed to cloud platforms (AWS, GCP, Azure)
- Orchestrated with Kubernetes
- Integrated into CI/CD pipelines
- Scaled horizontally for production loads

## Architecture Notes

### Multi-stage Build Benefits
- **Smaller image size**: Only production dependencies included
- **Better security**: Build tools not in final image
- **Faster deployments**: Smaller images transfer quicker

### Security Features
- **Non-root user**: Reduces attack surface
- **Read-only file system compatible**: Can run with `--read-only` flag
- **No unnecessary packages**: Slim base image

### Health Check Configuration
- **Interval**: 30 seconds between checks
- **Timeout**: 10 seconds max per check
- **Start period**: 40 seconds grace period for startup
- **Retries**: 3 failures before marking unhealthy