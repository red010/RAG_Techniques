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


class SimpleRAGGemini:
    """
    Classe per gestire il processo RAG semplice con chunking documenti e retrieval query usando embeddings Gemini.

    Questa classe incapsula l'intero pipeline RAG dalla pre-elaborazione documenti al retrieval query,
    fornendo un'interfaccia pulita per codificare documenti PDF ed eseguire query basate su similarità con Gemini.
    """

    def __init__(self, path, chunk_size=1000, chunk_overlap=200, n_retrieved=2):
        """
        Inizializza SimpleRAGGemini codificando il documento PDF e creando il retriever.

        Questo metodo esegue il pipeline core di pre-elaborazione documenti:
        1. Caricamento PDF ed estrazione testo usando PyPDFLoader
        2. Suddivisione testo in chunk gestibili con overlap specificato
        3. Pulizia testo per gestire problemi di formattazione
        4. Creazione embeddings vettoriali usando Gemini embeddings
        5. Creazione vector store Chroma per ricerca similarità efficiente
        6. Configurazione retriever per processamento query

        Args:
            path (str): Percorso al file PDF da codificare.
            chunk_size (int): Dimensione di ogni chunk di testo in caratteri (default: 1000).
                         Chunk più grandi preservano più contesto ma possono ridurre precisione retrieval.
            chunk_overlap (int): Numero di caratteri di overlap tra chunk consecutivi (default: 200).
                         Aiuta a mantenere continuità contesto tra confini chunk.
            n_retrieved (int): Numero di chunk più rilevanti da recuperare per ogni query (default: 2).
                         Più chunk forniscono contesto più ricco ma aumentano tempo processamento.
        """
        print("\n--- Inizializzazione Retriever RAG con Gemini ---")

        # Pre-elaborazione Documento: Codifica PDF in vector store usando embeddings Gemini
        # Questo coinvolge caricamento, chunking, pulizia e embedding del contenuto documento
        start_time = time.time()
        self.vector_store = encode_pdf(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.time_records = {'Chunking': time.time() - start_time}
        print(f"Tempo Chunking: {self.time_records['Chunking']:.2f} secondi")

        # Setup Retriever: Crea retriever configurato per recuperare top-k chunk più rilevanti
        # Il retriever usa Chroma per ricerca similarità efficiente nello spazio vettoriale
        self.chunks_query_retriever = self.vector_store.as_retriever(search_kwargs={"k": n_retrieved})

    def run(self, query):
        """
        Esegue la fase di retrieval del RAG per una query data usando embeddings Gemini.

        Questo metodo esegue l'operazione core di retrieval:
        1. Effettua embedding della query di input usando modello Gemini
        2. Esegue ricerca similarità nello spazio vettoriale Chroma
        3. Recupera i top-k chunk documenti più rilevanti
        4. Mostra il contesto recuperato per ispezione utente

        Args:
            query (str): La domanda o query dell'utente per cui recuperare contesto rilevante.
                        La query sarà embedded e confrontata con i chunk documenti.

        Returns:
            None: I risultati sono mostrati direttamente in console per feedback immediato.
        """
        # Fase Retrieval: Esegue ricerca similarità per trovare chunk documenti rilevanti
        # La query è embedded usando Gemini e confrontata con tutti i chunk documenti nello spazio vettoriale
        start_time = time.time()
        context = retrieve_context_per_question(query, self.chunks_query_retriever)
        self.time_records['Retrieval'] = time.time() - start_time
        print(f"Tempo Retrieval: {self.time_records['Retrieval']:.2f} secondi")

        # Mostra contesto recuperato: Visualizza i chunk più rilevanti trovati dal retriever
        # Questo permette agli utenti di ispezionare qualità e rilevanza delle informazioni recuperate
        show_context(context)


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Codifica un libro PDF in un vector store usando embeddings Gemini.

    Args:
        path: Il percorso al file PDF.
        chunk_size: La dimensione desiderata di ogni chunk di testo.
        chunk_overlap: La quantità di overlap tra chunk consecutivi.

    Returns:
        Un vector store Chroma contenente il contenuto del libro codificato.
    """

    # Carica documenti PDF
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Suddivide documenti in chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Crea embeddings Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Crea vector store con Chroma
    vectorstore = Chroma.from_documents(cleaned_texts, embeddings)

    return vectorstore


def validate_args(args):
    """
    Valida argomenti da linea di comando per assicurare che soddisfino i requisiti RAG.

    Questa funzione esegue validazione input per prevenire errori runtime e assicurare
    che i parametri di chunking e retrieval siano in range accettabili.

    Args:
        args: Argomenti da linea di comando parsati da argparse.

    Returns:
        args: Argomenti validati (restituiti invariati se validazione passa).

    Raises:
        ValueError: Se qualche argomento fallisce i criteri di validazione.
    """
    if args.chunk_size <= 0:
        raise ValueError("chunk_size deve essere un intero positivo.")
    if args.chunk_overlap < 0:
        raise ValueError("chunk_overlap deve essere un intero non-negativo.")
    if args.n_retrieved <= 0:
        raise ValueError("n_retrieved deve essere un intero positivo.")
    return args


def parse_args():
    """
    Parsea e valida argomenti da linea di comando per il sistema RAG Gemini semplice.

    Questa funzione configura il parser di argomenti con tutti i parametri configurabili
    per il pipeline RAG, fornendo valori di default ragionevoli permettendo personalizzazione completa.

    Returns:
        args: Argomenti da linea di comando validati pronti per l'uso.
    """
    parser = argparse.ArgumentParser(
        description="Codifica un documento PDF e testa un sistema RAG semplice con Gemini.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python 01_simple_rag_langchain_google.py --path data/document.pdf --query "Cos'è il machine learning?"
  python 01_simple_rag_langchain_google.py --chunk_size 500 --chunk_overlap 100 --n_retrieved 3 --evaluate
        """
    )

    parser.add_argument("--path", type=str, default="data/Understanding_Climate_Change.pdf",
                        help="Percorso al file PDF da codificare ed elaborare.")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Dimensione di ogni chunk di testo in caratteri (default: 1000). "
                             "Valori più grandi preservano più contesto ma possono ridurre precisione.")
    parser.add_argument("--chunk_overlap", type=int, default=200,
                        help="Overlap tra chunk consecutivi in caratteri (default: 200). "
                             "Aiuta a mantenere continuità contesto tra confini chunk.")
    parser.add_argument("--n_retrieved", type=int, default=2,
                        help="Numero di chunk più rilevanti da recuperare per query (default: 2). "
                             "Più chunk forniscono contesto più ricco ma aumentano tempo processamento.")
    parser.add_argument("--query", type=str, default="Qual è la causa principale del cambiamento climatico?",
                        help="Query per testare il retriever (default: domanda cambiamento climatico).")
    parser.add_argument("--evaluate", action="store_true",
                        help="Abilita valutazione prestazioni del sistema RAG (default: False). "
                             "Quando abilitato, esegue metriche di valutazione complete.")

    # Parsea e valida argomenti
    return validate_args(parser.parse_args())


def main(args):
    """
    Funzione di esecuzione principale che orchestra il completo pipeline RAG con Gemini.

    Questa funzione serve come punto di ingresso che:
    1. Inizializza il sistema SimpleRAGGemini con parametri utente specificati
    2. Esegue la query di retrieval
    3. Opzionalmente valuta le prestazioni del sistema

    Args:
        args: Argomenti da linea di comando parsati e validati.
    """
    # Inizializza il sistema RAG con codifica documento e setup retriever
    simple_rag = SimpleRAGGemini(
        path=args.path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        n_retrieved=args.n_retrieved
    )

    # Esegue la query di retrieval e mostra risultati
    simple_rag.run(args.query)

    # Fase di valutazione opzionale: Valuta qualità retrieval e metriche prestazioni
    if args.evaluate:
        print("\n--- Valutazione Prestazioni Sistema RAG ---")
        evaluate_rag(simple_rag.chunks_query_retriever)


if __name__ == '__main__':
    # Chiama la funzione main con argomenti parsati
    main(parse_args())


