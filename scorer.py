# scorer.py
import numpy as np
from openai import OpenAI

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # this loads your .env key for this file too
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed(text: str, model: str = "text-embedding-3-small"):
    # returns a numpy vector for cosine similarity
    v = client.embeddings.create(model=model, input=text).data[0].embedding
    return np.array(v, dtype=np.float32)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
