# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables
# These are defaults and can be overridden at runtime (e.g., by Cloud Run)
ENV PORT=8080
ENV FLASK_APP=app.py 
# For Gunicorn, if not using app.run() directly in prod
ENV PYTHONUNBUFFERED=TRUE 

# BGE M3 Model Configuration (can be overridden)
# If you download the model and add it to your Docker image,
# you can set MODEL_NAME to the local path within the container (e.g., /app/bge-m3-model)
ENV MODEL_NAME='BAAI/bge-m3' 
# Valid values for USE_FP16: 'true', 'false', or empty (default behavior based on device)
ENV USE_FP16='' 
# Valid values for MODEL_DEVICE: 'cpu', 'cuda:0', or empty (auto-detect)
ENV MODEL_DEVICE='' 

# API Keys - IMPORTANT: These should be set as secrets in Cloud Run, not hardcoded here.
# This Dockerfile just shows they are expected environment variables.
# Example, replace or set via Cloud Run secrets
ENV ALLOWED_API_KEYS="your_api_key_1,your_api_key_2" 
# Example, replace or set via Cloud Run secrets
ENV GEMINI_API_KEY="your_gemini_api_key" 

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Recommended: Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define the command to run the application
# Use Gunicorn for production. The number of workers can be tuned.
# Ensure 'app' in 'app:app' matches the Flask application instance variable name in your app.py
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "120", "app:app"]

# Healthcheck (optional but good for Cloud Run to know if app is ready)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
#   CMD curl -f http://localhost:8080/health || exit 1 