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
python 01_simple_rag_langchain_google.py
"""

import os
import sys
import time
from dotenv import load_dotenv

# =============================================================================
# CONFIGURAZIONE PARAMETRI RAG - MODIFICARE QUI PER PROVE DIVERSE
# =============================================================================

# 📄 DOCUMENTO TARGET: Percorso del file PDF da analizzare
# Modificare questo path per testare con documenti diversi
DOCUMENT_PATH = "data/Understanding_Climate_Change.pdf"

# 📏 DIMENSIONE CHUNK: Numero di caratteri per ogni frammento di testo
# Più grande = più contesto per chunk ma meno precisione nella ricerca
# Più piccolo = meno contesto ma ricerca più precisa
# Valore consigliato: 500-2000 caratteri
CHUNK_SIZE = 1000

# 🔗 SOVRAPPOSIZIONE CHUNK: Caratteri di overlap tra chunk consecutivi
# Mantiene continuità del contesto tra chunk adiacenti
# Valore tipico: 10-30% della dimensione chunk (qui 20%)
CHUNK_OVERLAP = 200

# 🎯 NUMERO CHUNK RECUPERATI: Quanti frammenti restituire per ogni domanda
# Più chunk = più informazioni contestuali ma risposta più lenta
# Meno chunk = risposta più veloce ma potenzialmente meno accurata
# Valore consigliato: 2-5 chunk
N_RETRIEVED = 3

# ❓ DOMANDA UTENTE: La query da sottoporre al sistema RAG
# Modificare questa stringa per testare domande diverse
USER_QUERY = "What is the main cause of climate change?"

# 📊 VALUTAZIONE: Abilita/disabilita la valutazione prestazioni del sistema
ENABLE_EVALUATION = False

# =============================================================================

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
        # 🏗️ SETUP RAG: Configurazione iniziale sistema RAG
        print("\n--- 🏗️ SETUP RAG: Configurazione sistema con Gemini ---")

        # 💾 CARICAMENTO DATI: Vector store con caching intelligente
        start_time = time.time()
        self.vector_store = load_or_create_vectorstore(path, chunk_size, chunk_overlap)
        self.time_records = {'VectorStore': time.time() - start_time}
        print(f"Tempo caricamento: {self.time_records['VectorStore']:.2f} secondi")

        # 🔍 CONFIGURAZIONE RETRIEVER: Setup motore di ricerca
        self.chunks_query_retriever = self.vector_store.as_retriever(search_kwargs={"k": n_retrieved})

    def run(self, query):
        """
        Esegue retrieval e mostra risultati.

        Args:
            query (str): Domanda utente.
        """
        # 🚀 CHIAMATA RAG: Esegue retrieval della domanda utente
        print(f"\n" + "="*60)
        print(f"🔍 DOMANDA UTENTE: {query}")
        print(f"="*60)

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


# Nota: I parametri sono ora definiti come costanti sopra per facilità di modifica
# Le funzioni CLI (parse_args, validate_args) sono mantenute in helper_functions.py per riuso


def main():
    """
    Esegue pipeline RAG completa con parametri configurati sopra.
    """
    # 🎯 PUNTO PRINCIPALE: Inizializzazione sistema RAG
    rag = SimpleRAGGemini(DOCUMENT_PATH, CHUNK_SIZE, CHUNK_OVERLAP, N_RETRIEVED)

    # 🚀 CHIAMATA RAG PRINCIPALE: Elabora la domanda dell'utente
    print(f"\n🤖 SISTEMA RAG ATTIVO - Elaborazione in corso...")
    rag.run(USER_QUERY)

    # 📊 VALUTAZIONE RAG (opzionale): Misura prestazioni del sistema
    if ENABLE_EVALUATION:
        print("\n--- 📈 VALUTAZIONE PRESTAZIONI RAG ---")
        evaluate_rag(rag.chunks_query_retriever)


# 🎬 PUNTO DI INGRESSO: Avvio esecuzione script RAG
if __name__ == '__main__':
    # 🚀 ESECUZIONE RAG: Avvia elaborazione completa con parametri configurati
    main()


