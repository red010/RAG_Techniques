from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from openai import RateLimitError
from typing import List
from rank_bm25 import BM25Okapi
import fitz
import asyncio
import random
import textwrap
import numpy as np
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


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Pipeline PDF → vector store OpenAI.

    Args:
        path: Percorso file PDF.
        chunk_size: Dimensione chunk caratteri.
        chunk_overlap: Sovrapposizione tra chunk.

    Returns:
        FAISS vector store con contenuto embedded.
    """

    # Caricamento PDF
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Chunking testo
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Embeddings e vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore


def encode_from_string(content, chunk_size=1000, chunk_overlap=200):
    """
    Pipeline testo → vector store OpenAI.

    Args:
        content (str): Testo da processare.
        chunk_size (int): Dimensione chunk.
        chunk_overlap (int): Sovrapposizione chunk.

    Returns:
        FAISS vector store con contenuto embedded.

    Raises:
        ValueError: Per input non validi.
        RuntimeError: Per errori durante encoding.
    """

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Content deve essere stringa non vuota.")

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size deve essere intero positivo.")

    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError("chunk_overlap deve essere intero non negativo.")

    try:
        # Chunking contenuto
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.create_documents([content])

        # Metadata per chunk
        for chunk in chunks:
            chunk.metadata['relevance_score'] = 1.0

        # Embeddings e vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

    except Exception as e:
        raise RuntimeError(f"Errore durante encoding: {str(e)}")

    return vectorstore


def retrieve_context_per_question(question, chunks_query_retriever):
    """
    Retrieval contesto per domanda.

    Args:
        question: Domanda utente.
        chunks_query_retriever: Retriever per ricerca similarità.

    Returns:
        Lista contenuti documenti rilevanti.
    """

    # Ricerca documenti rilevanti
    docs = chunks_query_retriever.invoke(question)

    # Estrazione contenuti
    context = [doc.page_content for doc in docs]

    return context


class QuestionAnswerFromContext(BaseModel):
    """
    Modello per risposta basata su contesto.

    Attributes:
        answer_based_on_content (str): Risposta generata dal contesto.
    """
    answer_based_on_content: str = Field(description="Genera risposta alla domanda basata sul contesto fornito.")


def create_question_answer_from_context_chain(llm):
    # Configurazione LLM per risposte contestuali
    question_answer_from_context_llm = llm

    # Template prompt per risposte basate su contesto
    question_answer_prompt_template = """
    Fornisci risposta concisa basata SOLO sul contesto fornito:
    {context}
    Domanda
    {question}
    """

    # Creazione prompt template
    question_answer_from_context_prompt = PromptTemplate(
        template=question_answer_prompt_template,
        input_variables=["context", "question"],
    )

    # Chain: prompt + LLM con output strutturato
    question_answer_from_context_cot_chain = question_answer_from_context_prompt | question_answer_from_context_llm.with_structured_output(
        QuestionAnswerFromContext)
    return question_answer_from_context_cot_chain


def answer_question_from_context(question, context, question_answer_from_context_chain):
    """
    Risponde domanda usando contesto fornito.

    Args:
        question: Domanda da rispondere.
        context: Contesto per risposta.
        question_answer_from_context_chain: Chain LLM per risposte.

    Returns:
        Dict con risposta, contesto e domanda.
    """
    input_data = {
        "question": question,
        "context": context
    }
    print("Risposta basata su contesto recuperato...")

    output = question_answer_from_context_chain.invoke(input_data)
    answer = output.answer_based_on_content
    return {"answer": answer, "context": context, "question": question}


def show_context(context):
    """
    Visualizza lista contesti recuperati.

    Args:
        context (list): Lista contesti da mostrare.

    Visualizza ogni contesto con numerazione.
    """
    for i, c in enumerate(context):
        print(f"Context {i + 1}:")
        print(c)
        print("\n")


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
