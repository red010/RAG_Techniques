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
import time
from dotenv import load_dotenv

# Aggiunge la directory genitore al path per accedere alle helper functions
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Carica variabili d'ambiente dal file .env
load_dotenv()
# Imposta la chiave API Google usando GEMINI_API_KEY dal file .env
os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY', '')

# Loader per documenti PDF
from langchain_community.document_loaders import PyPDFLoader

# Funzioni helper per RAG (provider embeddings, retrieval, pulizia testo, visualizzazione)
from helper_functions import (EmbeddingProvider,
                              retrieve_context_per_question,
                              replace_t_with_space,
                              get_langchain_embedding_provider,
                              get_file_hash,
                              encode_pdf,
                              load_or_create_vectorstore,
                              validate_args,  # Aggiunto validate_args
                              create_rag_parser,  # Aggiunto create_rag_parser
                              show_context)

# Funzione per valutazione prestazioni sistema RAG
from evaluation.evalute_rag import evaluate_rag

# Vector store Chroma per storage efficiente di embeddings
from langchain_chroma import Chroma


class SimpleRAGGemini:
    """
    Sistema RAG con Gemini: elabora PDF e risponde a query.
    """

    def __init__(self, path, chunk_size=1000, chunk_overlap=200, n_retrieved=2):
        """
        Inizializza RAG con caching intelligente.

        Args:
            path (str): Percorso PDF.
            chunk_size (int): Dimensione chunk.
            chunk_overlap (int): Overlap chunk.
            n_retrieved (int): Numero risultati retrieval.
        """
        print("\n--- Inizializzazione RAG con Gemini ---")

        # Carica/crea vector store con caching
        start_time = time.time()
        self.vector_store = load_or_create_vectorstore(path, chunk_size, chunk_overlap)
        self.time_records = {'VectorStore': time.time() - start_time}
        print(f"Tempo caricamento: {self.time_records['VectorStore']:.2f} secondi")

        # Configura retriever
        self.chunks_query_retriever = self.vector_store.as_retriever(search_kwargs={"k": n_retrieved})

    def run(self, query):
        """
        Esegue retrieval e mostra risultati.

        Args:
            query (str): Domanda utente.
        """
        # Retrieval con timing
        start_time = time.time()
        context = retrieve_context_per_question(query, self.chunks_query_retriever)
        self.time_records['Retrieval'] = time.time() - start_time
        print(f"Tempo retrieval: {self.time_records['Retrieval']:.2f} secondi")

        # Statistiche totali
        total_time = self.time_records.get('VectorStore', 0) + self.time_records['Retrieval']
        print(f"Tempo totale: {total_time:.2f} secondi")

        # Mostra risultati
        show_context(context)


def parse_args():
    """
    Crea e configura parser CLI per RAG.

    Returns:
        args: Argomenti validati.
    """
    parser = create_rag_parser()
    return validate_args(parser.parse_args())


def main(args):
    """
    Esegue pipeline RAG completa.

    Args:
        args: Parametri CLI.
    """
    # Inizializza ed esegui RAG
    rag = SimpleRAGGemini(args.path, args.chunk_size, args.chunk_overlap, args.n_retrieved)
    rag.run(args.query)

    # Valutazione opzionale
    if args.evaluate:
        print("\n--- Valutazione RAG ---")
        evaluate_rag(rag.chunks_query_retriever)


if __name__ == '__main__':
    main(parse_args())


