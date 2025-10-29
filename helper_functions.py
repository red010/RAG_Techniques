from openai import RateLimitError
from typing import List
from rank_bm25 import BM25Okapi
import fitz
import asyncio
import random
import textwrap
import numpy as np
import hashlib
import os
from enum import Enum


def replace_t_with_space(list_of_documents):
    """
    Sostituisce tab con spazi nei documenti.

    Args:
        list_of_documents: Lista documenti da pulire.

    Returns:
        Documenti con tab sostituiti da spazi.
    """

    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')  # Pulizia tab
    return list_of_documents


def text_wrap(text, width=120):
    """
    Formatta testo a larghezza fissa.

    Args:
        text (str): Testo da formattare.
        width (int): Larghezza massima righe.

    Returns:
        str: Testo formattato.
    """
    return textwrap.fill(text, width=width)


def get_file_hash(filepath):
    """
    Calcola hash SHA256 del file per verificare cambiamenti.

    Args:
        filepath (str): Percorso del file.

    Returns:
        str: Hash SHA256 del file.
    """
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()














def read_pdf_to_string(path):
    """
    Estrae testo completo da PDF.

    Args:
        path (str): Percorso file PDF.

    Returns:
        str: Testo concatenato di tutte le pagine.

    Usa PyMuPDF per estrarre testo da ogni pagina.
    """
    # Apertura PDF
    doc = fitz.open(path)
    content = ""
    # Iterazione pagine
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Estrazione testo pagina
        content += page.get_text()
    return content


def bm25_retrieval(bm25: BM25Okapi, cleaned_texts: List[str], query: str, k: int = 5) -> List[str]:
    """
    Retrieval BM25 per query.

    Args:
        bm25 (BM25Okapi): Indice BM25 precalcolato.
        cleaned_texts (List[str]): Lista testi puliti.
        query (str): Query di ricerca.
        k (int): Numero risultati top.

    Returns:
        List[str]: Top k testi basati su punteggi BM25.
    """
    # Tokenizzazione query
    query_tokens = query.split()

    # Calcolo punteggi BM25
    bm25_scores = bm25.get_scores(query_tokens)

    # Indici top k risultati
    top_k_indices = np.argsort(bm25_scores)[::-1][:k]

    # Recupero testi top k
    top_k_texts = [cleaned_texts[i] for i in top_k_indices]

    return top_k_texts


async def exponential_backoff(attempt):
    """
    Backoff esponenziale con jitter per retry.

    Args:
        attempt: Numero tentativo corrente.

    Attende periodo calcolato prima retry.
    Tempo: (2^tentativo) + frazione casuale secondi.
    """
    # Calcolo tempo attesa con backoff e jitter
    wait_time = (2 ** attempt) + random.uniform(0, 1)
    print(f"Rate limit superato. Retry tra {wait_time:.2f} secondi...")

    # Sleep asincrono
    await asyncio.sleep(wait_time)


async def retry_with_exponential_backoff(coroutine, max_retries=5):
    """
    Retry coroutine con backoff esponenziale su RateLimitError.

    Args:
        coroutine: Coroutine da eseguire.
        max_retries: Numero massimo tentativi.

    Returns:
        Risultato coroutine se riuscita.

    Raises:
        Ultima eccezione se tutti retry falliscono.
    """
    for attempt in range(max_retries):
        try:
            # Tentativo esecuzione coroutine
            return await coroutine
        except RateLimitError as e:
            # Se ultimo tentativo fallisce, rilancia eccezione
            if attempt == max_retries - 1:
                raise e

            # Attesa backoff esponenziale prima retry
            await exponential_backoff(attempt)

    # Se max retry raggiunti senza successo
    raise Exception("Max tentativi raggiunti")


# Provider embeddings disponibili
class EmbeddingProvider(Enum):
    OPENAI = "openai"
    COHERE = "cohere"
    AMAZON_BEDROCK = "bedrock"
    GOOGLE = "google"

# Provider modelli disponibili
class ModelProvider(Enum):
    OPENAI = "openai"
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    AMAZON_BEDROCK = "bedrock"
    GOOGLE = "google"


def get_langchain_embedding_provider(provider: EmbeddingProvider, model_id: str = None):
    """
    Factory provider embeddings LangChain.

    Args:
        provider (EmbeddingProvider): Provider da usare (OPENAI, COHERE, BEDROCK, GOOGLE).
        model_id (str): ID modello specifico (opzionale).

    Returns:
        Istanza provider embeddings LangChain.

    Raises:
        ValueError: Se provider non supportato.
    """
    if provider == EmbeddingProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    elif provider == EmbeddingProvider.COHERE:
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings()
    elif provider == EmbeddingProvider.AMAZON_BEDROCK:
        from langchain_community.embeddings import BedrockEmbeddings
        return BedrockEmbeddings(model_id=model_id) if model_id else BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    elif provider == EmbeddingProvider.GOOGLE:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=model_id) if model_id else GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    else:
        raise ValueError(f"Provider embeddings non supportato: {provider}")


def get_langchain_model_provider(provider: ModelProvider, model_id: str = None, temperature: float = 0.7):
    """
    Factory provider modelli LLM LangChain.

    Args:
        provider (ModelProvider): Provider da usare (OPENAI, GROQ, ANTHROPIC, BEDROCK, GOOGLE).
        model_id (str): ID modello specifico (opzionale).
        temperature (float): Temperatura per generazione (default: 0.7).

    Returns:
        Istanza modello LLM LangChain.

    Raises:
        ValueError: Se provider non supportato.
    """
    if provider == ModelProvider.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_id, temperature=temperature) if model_id else ChatOpenAI(temperature=temperature)
    elif provider == ModelProvider.GROQ:
        from langchain_groq import ChatGroq
        return ChatGroq(model=model_id, temperature=temperature) if model_id else ChatGroq(temperature=temperature)
    elif provider == ModelProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_id, temperature=temperature) if model_id else ChatAnthropic(temperature=temperature)
    elif provider == ModelProvider.AMAZON_BEDROCK:
        from langchain_community.chat_models import BedrockChat
        return BedrockChat(model_id=model_id, temperature=temperature) if model_id else BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0", temperature=temperature)
    elif provider == ModelProvider.GOOGLE:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_id, temperature=temperature) if model_id else ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)
    else:
        raise ValueError(f"Provider modello non supportato: {provider}")

