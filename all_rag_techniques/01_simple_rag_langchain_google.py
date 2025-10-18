"""
Sistema RAG (Retrieval-Augmented Generation) Semplice con Gemini

Panoramica:
Questo script implementa un sistema RAG di base per elaborare e interrogare documenti PDF.
Il sistema codifica il contenuto del documento in un vector store usando embeddings Gemini,
che può essere interrogato per recuperare informazioni rilevanti.

Componenti Chiave:
1. Elaborazione PDF ed estrazione testo
2. Suddivisione testo in chunk per elaborazione gestibile
3. Creazione vector store usando Chroma e embeddings Gemini
4. Configurazione retriever per interrogare i documenti elaborati
5. Valutazione del sistema RAG

Utilizzo:
python 01_simple_rag_langchain_google.py --path data/document.pdf --query "Qual è l'argomento principale?"
"""

import os
import sys
import argparse
import time
import hashlib
from dotenv import load_dotenv

# Aggiunge la directory genitore al path per accedere alle helper functions
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Carica variabili d'ambiente dal file .env
load_dotenv()
# Imposta la chiave API Google usando GEMINI_API_KEY dal file .env
os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY', '')

# Import per embeddings Gemini di Google
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Loader per documenti PDF
from langchain_community.document_loaders import PyPDFLoader

# Suddivisore di testo ricorsivo per chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Funzioni helper per RAG (provider embeddings, retrieval, pulizia testo, visualizzazione)
from helper_functions import (EmbeddingProvider,
                              retrieve_context_per_question,
                              replace_t_with_space,
                              get_langchain_embedding_provider,
                              show_context)

# Funzione per valutazione prestazioni sistema RAG
from evaluation.evalute_rag import evaluate_rag

# Vector store Chroma per storage efficiente di embeddings
from langchain_community.vectorstores import Chroma


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


def load_or_create_vectorstore(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Carica vector store esistente o ne crea uno nuovo se necessario.

    Args:
        pdf_path (str): Percorso al file PDF.
        chunk_size (int): Dimensione chunk.
        chunk_overlap (int): Overlap tra chunk.

    Returns:
        Chroma: Vector store caricato o creato.
    """
    # Crea directory per vector stores se non esiste
    persist_dir = os.path.join(os.path.dirname(pdf_path), ".vector_stores")
    os.makedirs(persist_dir, exist_ok=True)

    # Calcola hash del file per identificare versioni
    file_hash = get_file_hash(pdf_path)
    collection_name = f"pdf_{file_hash[:16]}"  # Nome collezione basato su hash
    vectorstore_path = os.path.join(persist_dir, collection_name)

    # Verifica se vector store esiste già
    if os.path.exists(vectorstore_path):
        try:
            print(f"✅ Carico vector store esistente per {os.path.basename(pdf_path)}")
            vectorstore = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            )
            # Verifica che abbia documenti
            if vectorstore._collection.count() > 0:
                print(f"✅ Vector store caricato: {vectorstore._collection.count()} documenti")
                return vectorstore
            else:
                print("⚠️ Vector store vuoto, ricreo...")
        except Exception as e:
            print(f"⚠️ Errore caricamento vector store esistente: {e}, ricreo...")

    # Crea nuovo vector store con persistenza automatica
    print(f"🔄 Creo nuovo vector store per {os.path.basename(pdf_path)}")
    documents = encode_pdf(pdf_path, chunk_size, chunk_overlap)
    vectorstore = Chroma.from_documents(
        documents,
        GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory=vectorstore_path
    )
    print(f"✅ Vector store creato: {vectorstore._collection.count()} documenti")

    return vectorstore


class SimpleRAGGemini:
    """
    Questa classe incapsula l'intera pipeline RAG dalla pre-elaborazione documenti al retrieval query
    """

    def __init__(self, path, chunk_size=1000, chunk_overlap=200, n_retrieved=2):
        """
        Pipeline RAG: caricamento PDF → chunking → pulizia testo → embeddings Gemini → vector store Chroma → retriever.

        Args:
            path (str): Percorso file PDF da processare.
            chunk_size (int): Dimensione chunk in caratteri (default: 1000). Più grandi = più contesto ma meno precisione.
            chunk_overlap (int): Overlap tra chunk in caratteri (default: 200). Mantiene continuità contesto.
            n_retrieved (int): Numero chunk da recuperare per query (default: 2). Più chunk = più contesto ma più lento.
        """
        print("\n--- Inizializzazione Retriever RAG con Gemini ---")

        # Carica vector store esistente o creane uno nuovo se necessario
        start_time = time.time()
        self.vector_store = load_or_create_vectorstore(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.time_records = {'VectorStore': time.time() - start_time}
        print(f"Tempo caricamento Vector Store: {self.time_records['VectorStore']:.2f} secondi")

        # Setup retriever per ricerca similarità in Chroma
        self.chunks_query_retriever = self.vector_store.as_retriever(search_kwargs={"k": n_retrieved})

    def run(self, query):
        """
        Retrieval RAG: embedding query → ricerca similarità Chroma → top-k risultati → visualizzazione contesto.

        Args:
            query (str): Domanda utente da processare per retrieval.

        Returns:
            None: Risultati mostrati in console.
        """
        # Retrieval: cerca chunk rilevanti usando embeddings Gemini
        start_time = time.time()
        context = retrieve_context_per_question(query, self.chunks_query_retriever)
        self.time_records['Retrieval'] = time.time() - start_time
        print(f"Tempo Retrieval: {self.time_records['Retrieval']:.2f} secondi")

        # Statistiche complessive
        total_time = self.time_records.get('VectorStore', 0) + self.time_records['Retrieval']
        print(f"Tempo totale: {total_time:.2f} secondi")

        # Mostra risultati retrieval per ispezione utente
        show_context(context)


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Pipeline encoding: PDF → chunking → pulizia testo.

    Args:
        path: Percorso file PDF.
        chunk_size: Dimensione chunk in caratteri.
        chunk_overlap: Overlap tra chunk consecutivi.

    Returns:
        Documenti processati e puliti (pronti per embeddings).
    """
    # Carica e processa PDF
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Chunking del testo
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    return cleaned_texts


def validate_args(args):
    """
    Valida parametri input per prevenire errori runtime.

    Args:
        args: Argomenti da linea di comando.

    Returns:
        args: Parametri validati.

    Raises:
        ValueError: Per parametri non validi.
    """
    if args.chunk_size <= 0:
        raise ValueError("chunk_size deve essere > 0.")
    if args.chunk_overlap < 0:
        raise ValueError("chunk_overlap deve essere >= 0.")
    if args.n_retrieved <= 0:
        raise ValueError("n_retrieved deve essere > 0.")
    return args


def parse_args():
    """
    Configura e parsea argomenti CLI per il sistema RAG Gemini.

    Returns:
        args: Argomenti validati pronti per l'uso.
    """
    parser = argparse.ArgumentParser(
        description="Sistema RAG con Gemini per processamento PDF.",
        epilog="""
Esempi:
  python 01_simple_rag_langchain_google.py --path data/document.pdf --query "Cos'è il machine learning?"
  python 01_simple_rag_langchain_google.py --chunk_size 500 --chunk_overlap 100 --n_retrieved 3 --evaluate
        """
    )

    parser.add_argument("--path", type=str, default="data/Understanding_Climate_Change.pdf",
                        help="Percorso file PDF da processare.")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Dimensione chunk (default: 1000). Più grandi = più contesto ma meno precisione.")
    parser.add_argument("--chunk_overlap", type=int, default=200,
                        help="Overlap chunk (default: 200). Mantiene continuità contesto.")
    parser.add_argument("--n_retrieved", type=int, default=2,
                        help="Chunk da recuperare (default: 2). Più chunk = più contesto ma più lento.")
    parser.add_argument("--query", type=str, default="Qual è la causa principale del cambiamento climatico?",
                        help="Query di test (default: domanda clima).")
    parser.add_argument("--evaluate", action="store_true",
                        help="Abilita valutazione prestazioni RAG.")

    return validate_args(parser.parse_args())


def main(args):
    """
    Punto di ingresso principale: inizializza RAG → esegue query → valuta prestazioni (opzionale).

    Args:
        args: Parametri CLI validati.
    """
    # Inizializza sistema RAG
    simple_rag = SimpleRAGGemini(
        path=args.path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        n_retrieved=args.n_retrieved
    )

    # Esegue retrieval e mostra risultati
    simple_rag.run(args.query)

    # Valutazione prestazioni opzionale
    if args.evaluate:
        print("\n--- Valutazione Prestazioni Sistema RAG ---")
        evaluate_rag(simple_rag.chunks_query_retriever)


if __name__ == '__main__':
    # Chiama la funzione main con argomenti parsati
    main(parse_args())


