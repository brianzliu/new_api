import requests
import json
import os

# --- Configuration ---
BASE_URL = "http://localhost:8080"  # Change if your API is hosted elsewhere

# API Key for BGE M3 model endpoints (/encode, /colbert-similarity)
# You can set this as an environment variable 'MY_CLIENT_API_KEY' 
# or replace the placeholder directly in the script.
# This key must be one of the keys defined in the server's ALLOWED_API_KEYS.
BGE_M3_API_KEY = os.getenv("MY_CLIENT_API_KEY", "your_dev_api_key1_for_client") 

# --- Helper Functions ---
def print_response(response):
    """Helper to print status code and JSON response."""
    print(f"Status Code: {response.status_code}")
    try:
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
    except json.JSONDecodeError:
        print("Response Text (or non-JSON stream):")
        print(response.text)
    print("-" * 40)

# --- Example API Calls ---

def example_get_root():
    print(">>> Testing / (root) endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print_response(response)
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        print("-" * 40)

def example_hello():
    print(">>> Testing /api/hello endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/hello")
        print_response(response)
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        print("-" * 40)

def example_gemini_test():
    print(">>> Testing /gemini/test endpoint...")
    # This endpoint relies on GEMINI_API_KEY being set on the server.
    try:
        response = requests.get(f"{BASE_URL}/gemini/test")
        print_response(response)
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        print("-" * 40)

def example_compare_papers(pdf_path1, pdf_path2):
    print(f">>> Testing /api/compare-papers endpoint with '{pdf_path1}' and '{pdf_path2}'...")
    if not (os.path.exists(pdf_path1) and os.path.isfile(pdf_path1)):
        print(f"Error: PDF file not found or is not a file: {pdf_path1}")
        print("Please create a dummy PDF file named 'dummy1.pdf' or update the path.")
        print("-" * 40)
        return
    if not (os.path.exists(pdf_path2) and os.path.isfile(pdf_path2)):
        print(f"Error: PDF file not found or is not a file: {pdf_path2}")
        print("Please create a dummy PDF file named 'dummy2.pdf' or update the path.")
        print("-" * 40)
        return

    files = None
    try:
        files = {
            'file1': (os.path.basename(pdf_path1), open(pdf_path1, 'rb'), 'application/pdf'),
            'file2': (os.path.basename(pdf_path2), open(pdf_path2, 'rb'), 'application/pdf')
        }
        
        print("Streaming response for /api/compare-papers (Ctrl+C to stop if it hangs on error):")
        with requests.post(f"{BASE_URL}/api/compare-papers", files=files, stream=True, timeout=300) as r:
            # Check for non-200 status before trying to iterate lines
            if r.status_code != 200:
                print(f"Error from server: Status Code {r.status_code}")
                print("Response content:")
                print(r.text) # Print the error response from the server
                print("-" * 40)
                return

            for line in r.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        # Print each JSON object from the stream
                        print(json.dumps(json.loads(decoded_line), indent=2))
                    except json.JSONDecodeError:
                        print(f"Raw line (not JSON): {decoded_line}")
        print("\nStream finished.")
        print("-" * 40)
    except requests.exceptions.HTTPError as e_http:
        print(f"HTTP Error: {e_http.response.status_code}")
        print("Response Text:")
        print(e_http.response.text)
        print("-" * 40)
    except requests.exceptions.RequestException as e_req:
        print(f"Request Error: {e_req}")
        print("-" * 40)
    except Exception as e_gen:
        print(f"An unexpected error occurred: {e_gen}")
        print("-" * 40)
    finally:
        if files:
            if files['file1'] and hasattr(files['file1'][1], 'close'):
                files['file1'][1].close()
            if files['file2'] and hasattr(files['file2'][1], 'close'):
                files['file2'][1].close()

def example_encode_texts():
    print(">>> Testing /encode endpoint...")
    if not BGE_M3_API_KEY or BGE_M3_API_KEY == "your_dev_api_key1_for_client":
        print("Warning: BGE_M3_API_KEY is not set (or is using a placeholder) in this script or as MY_CLIENT_API_KEY environment variable.")
        print("The /encode call might fail if the API key is required and not correctly configured on the server's ALLOWED_API_KEYS list.")

    headers = {
        "X-API-Key": BGE_M3_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "texts": [
            "This is the first sentence for encoding.",
            "And here is another sentence.",
            "The BGE-M3 model is versatile."
        ],
        "return_dense": True,
        "return_sparse": False, # Set to true if you want sparse vectors
        "return_colbert_vecs": True, # Set to true for ColBERT vectors
        "compute_colbert_pairwise_scores": True # Requires return_colbert_vecs=True and at least 2 texts
    }
    try:
        response = requests.post(f"{BASE_URL}/encode", headers=headers, json=payload, timeout=60)
        print_response(response)
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        print("-" * 40)

def example_colbert_similarity():
    print(">>> Testing /colbert-similarity endpoint...")
    if not BGE_M3_API_KEY or BGE_M3_API_KEY == "your_dev_api_key1_for_client":
        print("Warning: BGE_M3_API_KEY is not set (or is using a placeholder) in this script or as MY_CLIENT_API_KEY environment variable.")
        print("The /colbert-similarity call might fail if the API key is required and not correctly configured on the server's ALLOWED_API_KEYS list.")

    headers = {
        "X-API-Key": BGE_M3_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "query_text": "What are the applications of NLP?",
        "candidate_texts": [
            "Natural Language Processing is used in machine translation.",
            "Topic modeling is a common NLP task.",
            "Quantum computing is a different field of study."
        ]
    }
    try:
        response = requests.post(f"{BASE_URL}/colbert-similarity", headers=headers, json=payload, timeout=60)
        print_response(response)
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        print("-" * 40)

if __name__ == "__main__":
    print("Running API Client Examples...")
    print(f"Using API Base URL: {BASE_URL}")
    print(f"Using BGE M3 API Key for client: {'Set (see value in script or MY_CLIENT_API_KEY env var)' if BGE_M3_API_KEY and BGE_M3_API_KEY != 'your_dev_api_key1_for_client' else 'NOT SET or using placeholder (edit script or set MY_CLIENT_API_KEY env var)'}")
    print("="*50)
    
    # Call example functions
    example_get_root()
    example_hello()
    example_gemini_test()
    
    # --- For /api/compare-papers ---
    # 1. Create two dummy PDF files (e.g., dummy1.pdf, dummy2.pdf) in the same directory as this script.
    #    You can create simple text files and save them as PDFs, or use any small PDF files for testing.
    # 2. Uncomment the line below to run the example.
    print("\nInstructions for /api/compare-papers example:")
    print("1. Create two PDF files (e.g., 'dummy1.pdf' and 'dummy2.pdf') in the same directory as this script.")
    print("2. Update the paths in the example_compare_papers function call below if needed.")
    print("3. Uncomment the function call to run.")
    # example_compare_papers("dummy1.pdf", "dummy2.pdf")
    # Or, provide full paths:
    # example_compare_papers("/path/to/your/first.pdf", "/path/to/your/second.pdf")
    print("-" * 40)
    
    example_encode_texts()
    example_colbert_similarity()

    print("\n" + "="*50)
    print("Finished running examples.")
    print("Important:")
    print("- Ensure your combined_api server (app.py) is running and accessible at the BASE_URL.")
    print("- For /encode and /colbert-similarity, ensure the BGE_M3_API_KEY used in this script is valid and present in the ALLOWED_API_KEYS on the server.")
    print("- For /gemini/test and /api/compare-papers, ensure GEMINI_API_KEY is correctly set as an environment variable on the server.")
    print("- If you haven't installed the 'requests' library in your Python environment, run: pip install requests") 