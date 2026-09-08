"""
llm.py — Existing LLM integration (unchanged).

This module contains the generate_answer() function that sends a prompt
to Google Gemini and returns the generated response. The RAG pipeline
imports and calls this function as-is.
"""

import os

from google import genai
from dotenv import load_dotenv

load_dotenv()


def get_client():
    """Get Gemini client if a valid API key is present."""
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    api_key = api_key.strip().strip("'").strip('"')
    if not api_key or api_key == "your_gemini_api_key_here":
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"[llm] Error initializing Gemini client: {e}")
        return None


import time
from config import LLM_MODELS

def generate_answer(context: str, question: str) -> str:
    """
    Send a question to Gemini along with retrieved document context.
    Attempts generation across a hierarchy of fallback models in case of
    temporary high demand (503), rate limits (429), or unavailable endpoints.
    """
    client = get_client()
    if not client:
        return "⚠️ **GEMINI_API_KEY is not configured or invalid.**\n\nPlease add your valid Gemini API Key to the `.env` file (`GEMINI_API_KEY=AIzaSy...`) to enable LLM answer generation."

    prompt = f"""You are CampusAI, a college information assistant. You answer questions STRICTLY and ONLY using the retrieved document context provided below.

STRICT RULES — YOU MUST FOLLOW ALL OF THESE:

1. ONLY use information that is EXPLICITLY stated in the Context below. Do NOT use your own knowledge, training data, or any external information.

2. Do NOT infer, assume, guess, or extrapolate beyond what the Context explicitly says.

3. Do NOT add opinions, recommendations, analysis, or commentary of your own.

4. Do NOT paraphrase in a way that changes the meaning of the source material. Stay faithful to the original text.

5. If the Context does NOT contain enough information to answer the question, you MUST respond EXACTLY with:
   "I could not find this information in the available college documents. Please contact the college administration for assistance."

6. Do NOT hallucinate or invent any college-specific facts, policies, names, dates, numbers, or procedures.

7. When answering, reference the source where you found the information (e.g., "According to the document...").

8. Try to respond casually when greeted and if you cannot find anything, kindly redirect the users to manually contact someone.
Context:
{context}

Question:
{question}

Answer:
    """

    models_to_try = LLM_MODELS if isinstance(LLM_MODELS, list) and LLM_MODELS else ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
    errors = []

    for model_name in models_to_try:
        try:
            print(f"[llm] Attempting generation with model: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if response and response.text:
                return response.text
        except Exception as e:
            err_msg = str(e)
            print(f"[llm] Fallback triggered: Model '{model_name}' failed with error: {err_msg}")
            errors.append(f"{model_name}: {err_msg}")
            time.sleep(0.3)  # brief pause before attempting fallback model
            continue

    return f"⚠️ All Gemini models encountered errors:\n" + "\n".join(f"- {err}" for err in errors)


