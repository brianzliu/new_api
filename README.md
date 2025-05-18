
## Prerequisites

*   Python 3.10+
*   Docker (for containerized deployment)
*   Google Cloud SDK (gcloud CLI, for deployment to Cloud Run)

## Local Development

1.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Copy `.env.example` to a new file named `.env`:
    ```bash
    cp .env.example .env
    ```
    Edit `.env` and fill in your actual `GEMINI_API_KEY` and any `ALLOWED_API_KEYS` you wish to use for development.
    The `ALLOWED_API_KEYS` are for the BGE M3 model endpoints (`/encode`, `/colbert-similarity`). If you leave `ALLOWED_API_KEYS` empty in your `.env` file, and the default in `app.py` is an empty set, these endpoints will be open (no API key required for local dev).

4.  **Run the application:**
    The Flask development server can be used for local testing:
    ```bash
    python app.py
    ```
    The API will be available at `http://localhost:8080` (or the port specified in your `.env` file).

    For a more production-like local run using Gunicorn (as defined in the Dockerfile CMD):
    ```bash
    gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120 app:app 
    ```
    (Ensure `$PORT` is set in your environment or replace it, e.g., `8080`)


## API Endpoints

### General

*   `GET /`: Welcome message.
*   `GET /health`: Health check endpoint. Shows status of BGE model and Gemini configuration.

### Backend/Gemini Endpoints

*   `GET /api/hello`: Simple hello message.
*   `GET /gemini/test`: Test endpoint for Gemini API.
*   `POST /api/compare-papers`: Compares two uploaded PDF files and streams a similarity analysis.
    *   **Form Data:** `file1` (PDF), `file2` (PDF)
    *   **Response:** Streams JSON objects with status and chunks of the analysis. Mimetype `application/json-seq`.

### BGE-M3 Model Endpoints

These endpoints require an `X-API-Key` header if `ALLOWED_API_KEYS` is configured.

*   `POST /encode`: Encodes a list of texts using the BGE-M3 model.
    *   **JSON Body:**
        ```json
        {
            "texts": ["text1", "text2"],
            "return_dense": true,       // optional, default true
            "return_sparse": false,     // optional, default false
            "return_colbert_vecs": false, // optional, default false
            "compute_colbert_pairwise_scores": false // optional, default false. If true and return_colbert_vecs is true, scores texts[1:] against texts[0]
        }
        ```
    *   **Response:** JSON with embeddings.

*   `POST /colbert-similarity`: Computes ColBERT similarity scores between a query and candidate texts.
    *   **JSON Body:**
        ```json
        {
            "query_text": "my query string",
            "candidate_texts": ["candidate 1", "candidate 2"]
        }
        ```
    *   **Response:** JSON with scores.


## Docker Build

To build the Docker image locally:
```bash
docker build -t combined-api:latest .
```
(Run this command from within the `combined_api` directory)

To run the Docker container locally (example):
```bash
docker run -p 8080:8080 \\
    -e PORT=8080 \\
    -e GEMINI_API_KEY=\"your_actual_gemini_key\" \\
    -e ALLOWED_API_KEYS=\"your_api_key\" \\
    # -e MODEL_NAME=\"BAAI/bge-m3\" # Or path if model is local in image
    combined-api:latest
```
Remember to replace placeholder API keys.

## Deployment to Google Cloud Run

1.  **Enable APIs:**
    Ensure you have the Cloud Run API, Artifact Registry API (or Container Registry API), and Cloud Build API enabled in your Google Cloud project.

2.  **Authenticate gcloud:**
    ```bash
    gcloud auth login
    gcloud config set project YOUR_PROJECT_ID
    ```

3.  **Configure Artifact Registry (if not already done):**
    Create a Docker repository in Artifact Registry:
    ```bash
    gcloud artifacts repositories create combined-api-repo \\
        --repository-format=docker \\
        --location=YOUR_REGION \\
        --description=\"Docker repository for Combined API\"
    ```
    Example: `YOUR_REGION` could be `us-central1`.

4.  **Build and Push the image using Cloud Build:**
    From the `combined_api` directory, submit a build to Cloud Build. This command builds the image and pushes it to your Artifact Registry.
    ```bash
    gcloud builds submit --tag YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/combined-api-repo/combined-api:latest .
    ```
    Replace `YOUR_REGION` and `YOUR_PROJECT_ID`.

5.  **Deploy to Cloud Run:**
    ```bash
    gcloud run deploy combined-api-service \\
        --image YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/combined-api-repo/combined-api:latest \\
        --platform managed \\
        --region YOUR_REGION \\
        --allow-unauthenticated \\ # Or configure authentication
        --port 8080 \\ # Port your container listens on (matches Dockerfile EXPOSE and Gunicorn bind)
        --set-env-vars GEMINI_API_KEY=\"YOUR_ACTUAL_GEMINI_KEY\" \\ # Set as secret in production!
        --set-env-vars ALLOWED_API_KEYS=\"YOUR_BGE_API_KEY\" \\     # Set as secret in production!
        # --set-env-vars MODEL_NAME=\"BAAI/bge-m3\" \\ # If using default HuggingFace model
        # --set-env-vars USE_FP16=\"true\" \\ # If deploying to GPU instances and want FP16
        # --set-env-vars MODEL_DEVICE=\"cuda:0\" \\ # If deploying to GPU instances
        --memory 2Gi \\ # Adjust as needed
        --cpu 1       # Adjust as needed
        --timeout 300s # Max request timeout, adjust as needed (Gunicorn timeout is 120s)
        --concurrency 80 # Adjust based on expected load and instance capabilities
    ```
    **IMPORTANT FOR PRODUCTION:**
    *   For `GEMINI_API_KEY` and `ALLOWED_API_KEYS`, use Cloud Run's support for **secrets** (e.g., integrate with Secret Manager) instead of passing them directly as environment variables in the deploy command for better security.
    *   Adjust memory, CPU, timeout, and concurrency settings based on your needs and the performance of the BGE-M3 model loading and inference. The BGE-M3 model can be memory-intensive.
    *   If you need GPU support on Cloud Run for the BGE-M3 model, you'll need to configure that when deploying (select a GPU instance type) and ensure your `MODEL_DEVICE` is set appropriately (e.g., `cuda:0`).

6.  **Using a local BGE-M3 Model on Cloud Run:**
    If you want to avoid downloading the BGE-M3 model on every container startup, you can:
    *   Download the model (e.g., from Hugging Face: `git lfs install && git clone https://huggingface.co/BAAI/bge-m3`).
    *   Create a directory in `combined_api` (e.g., `bge_model_files`) and place the model files there.
    *   Modify your `Dockerfile` to `COPY bge_model_files /app/bge_model_files` (or your chosen path).
    *   When deploying to Cloud Run, set the `MODEL_NAME` environment variable to the path within the container, e.g., `--set-env-vars MODEL_NAME="/app/bge_model_files"`.
    *   This will increase your Docker image size but can speed up cold starts.
