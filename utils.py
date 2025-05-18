import io
from PyPDF2 import PdfReader
import google.generativeai as genai
#from dotenv import load_dotenv # Removed
import os
import json

#load_dotenv() # Removed

#genai.configure(api_key=os.getenv("GEMINI_API_KEY")) # Removed

# for similarity paragraph generation
def extract_text_from_pdf(pdf_bytes):
    pdf_file = io.BytesIO(pdf_bytes)
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def generate_similarity_paragraph_stream(pdf_bytes1, pdf_bytes2):
    """Generate a similarity analysis paragraph between two PDFs with streaming response"""
    import PyPDF2
    import io
    
    # Extract text from PDFs
    def extract_text_from_pdf(pdf_bytes):
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    text1 = extract_text_from_pdf(pdf_bytes1)
    text2 = extract_text_from_pdf(pdf_bytes2)
    
    # Truncate texts if they're too long to fit in the context
    max_chars = 10000  # Adjust based on Gemini's limitations
    if len(text1) > max_chars:
        text1 = text1[:max_chars] + "... [truncated]"
    if len(text2) > max_chars:
        text2 = text2[:max_chars] + "... [truncated]"
    
    # Create a prompt for Gemini
    prompt = f"""
    I have two research papers. Here are extracts from both:
    
    PAPER 1:
    {text1}
    
    PAPER 2:
    {text2}
    
    Compare these two research papers and analyze their similarities and differences in terms of:
    1. Research topics and focus areas
    2. Methodologies used
    3. Key findings and conclusions
    4. Potential areas of complementarity or contradiction
    
    Write a comprehensive paragraph that explains how these papers are related to each other.
    """
    
    # Generate content using Gemini with streaming
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    response = model.generate_content(prompt, stream=True)
    
    for chunk in response:
        if chunk.text:
            yield chunk.text 