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
# USER_QUERY = "How does deforestation in tropical regions affect biodiversity?"
# USER_QUERY = "How does agriculture affect the climate?"
# USER_QUERY = "Who is most vulnerable to climatic shifts?"
# USER_QUERY = "Describe non-technological climate solutions."
# USER_QUERY = "Explain the nexus of climate, biodiversity, and health."

# 📊 VALUTAZIONE: Abilita/disabilita la valutazione prestazioni del sistema
ENABLE_EVALUATION = True

# =============================================================================

# Aggiunge la directory genitore al path per accedere alle helper functions
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Carica variabili d'ambiente dal file .env
load_dotenv()
# Imposta la chiave API Google usando GEMINI_API_KEY dal file .env
os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY', '')

# Loader e splitter per documenti PDF
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings e vector store
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Utility per hashing e timing
import hashlib

# Funzioni helper per RAG (solo factory e utility generiche)
from helper_functions import (EmbeddingProvider,
                              ModelProvider,
                              get_langchain_embedding_provider,
                              get_langchain_model_provider,
                              get_file_hash)

# Funzione per valutazione prestazioni sistema RAG
from evaluation.evalute_rag import evaluate_rag


class SimpleRAGGemini:
    """
    Sistema RAG con Gemini: elabora PDF e risponde a query.
    """

    def __init__(self, path, chunk_size=1000, chunk_overlap=200, n_retrieved=2):
        """
        Inizializza RAG con pipeline completa esplicita.

        Args:
            path (str): Percorso PDF.
            chunk_size (int): Dimensione chunk.
            chunk_overlap (int): Overlap chunk.
            n_retrieved (int): Numero risultati retrieval.
        """
        # 🏗️ SETUP RAG: Configurazione iniziale sistema RAG
        print("\n" + "="*70)
        print("🏗️ SETUP RAG: Configurazione sistema con Gemini")
        print("="*70)
        
        # 1/4 - CARICAMENTO DOCUMENTO
        print("\n📄 FASE 1/4 - Caricamento documento PDF")
        print("-" * 70)
        start_time = time.time()
        loader = PyPDFLoader(path)
        documents = loader.load()
        load_time = time.time() - start_time
        print(f"   ✓ Documento caricato: {len(documents)} pagine")
        print(f"   ⏱️  Tempo: {load_time:.2f}s")
        
        # 2/4 - SPLITTING TESTO
        print("\n✂️  FASE 2/4 - Suddivisione in chunk")
        print("-" * 70)
        start_time = time.time()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Pulizia tab dal contenuto
        for doc in chunks:
            doc.page_content = doc.page_content.replace('\t', ' ')
        
        split_time = time.time() - start_time
        print(f"   ✓ Chunk creati: {len(chunks)}")
        print(f"   📊 Parametri: size={chunk_size}, overlap={chunk_overlap}")
        print(f"   ⏱️  Tempo: {split_time:.2f}s")
        
        # 3/4 - VETTORIZZAZIONE E STORAGE
        print("\n🧮 FASE 3/4 - Creazione embeddings e vector store")
        print("-" * 70)
        start_time = time.time()
        
        # Setup caching intelligente basato su hash del file e parametri
        persist_dir = os.path.join(os.path.dirname(path), ".vector_stores")
        os.makedirs(persist_dir, exist_ok=True)
        
        file_hash = get_file_hash(path)
        vectorstore_path = os.path.join(
            persist_dir, 
            f"pdf_{file_hash[:8]}_c{chunk_size}_o{chunk_overlap}_goog"
        )
        
        # Inizializza embeddings Google Gemini
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Carica da cache o crea nuovo vector store
        if os.path.exists(vectorstore_path):
            try:
                self.vector_store = Chroma(
                    persist_directory=vectorstore_path,
                    embedding_function=embeddings
                )
                # Verifica che la cache sia valida
                if self.vector_store._collection.count() > 0:
                    print(f"   ✓ Vector store caricato da cache")
                    print(f"   📦 Documenti in cache: {self.vector_store._collection.count()}")
                else:
                    raise Exception("Cache vuota")
            except Exception as e:
                print(f"   ⚠️  Cache invalida, ricreo vector store...")
                self.vector_store = Chroma.from_documents(
                    chunks, embeddings, persist_directory=vectorstore_path
                )
                print(f"   ✓ Vector store creato e salvato")
                print(f"   📦 Documenti indicizzati: {self.vector_store._collection.count()}")
        else:
            print(f"   🆕 Nessuna cache trovata, creo nuovo vector store...")
            self.vector_store = Chroma.from_documents(
                chunks, embeddings, persist_directory=vectorstore_path
            )
            print(f"   ✓ Vector store creato e salvato")
            print(f"   📦 Documenti indicizzati: {self.vector_store._collection.count()}")
        
        store_time = time.time() - start_time
        print(f"   ⏱️  Tempo: {store_time:.2f}s")
        
        # 4/4 - CONFIGURAZIONE RETRIEVER
        print("\n🔍 FASE 4/4 - Configurazione retriever")
        print("-" * 70)
        self.chunks_query_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": n_retrieved}
        )
        print(f"   ✓ Retriever configurato")
        print(f"   🎯 Top-K documenti: {n_retrieved}")
        
        # Riepilogo finale
        total_time = load_time + split_time + store_time
        print("\n" + "="*70)
        print(f"✅ SETUP COMPLETATO")
        print(f"   ⏱️  Tempo totale: {total_time:.2f}s")
        print("="*70 + "\n")

    def run(self, query):
        """
        Esegue retrieval e mostra risultati.

        Args:
            query (str): Domanda utente.
        """
        # 🚀 CHIAMATA RAG: Esegue retrieval della domanda utente
        print("\n" + "="*70)
        print(f"🔍 QUERY UTENTE")
        print("="*70)
        print(f"❓ {query}")
        print("="*70 + "\n")

        # RETRIEVAL - Ricerca documenti rilevanti
        print("🔎 Ricerca chunk rilevanti nel vector store...")
        start_time = time.time()
        retrieved_docs = self.chunks_query_retriever.invoke(query)
        retrieval_time = time.time() - start_time
        print(f"   ✓ {len(retrieved_docs)} chunk recuperati in {retrieval_time:.2f}s\n")

        # VISUALIZZAZIONE RISULTATI
        print("="*70)
        print(f"📄 CHUNK RECUPERATI ({len(retrieved_docs)} risultati)")
        print("="*70 + "\n")
        
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"{'─'*70}")
            print(f"📌 CHUNK #{i}")
            print(f"{'─'*70}")
            print(doc.page_content)
            
            # Mostra metadata se disponibili
            if doc.metadata:
                print(f"\n📊 Metadata:")
                for key, value in doc.metadata.items():
                    print(f"   • {key}: {value}")
            print("\n")
        
        # Riepilogo finale
        print("="*70)
        print(f"✅ RETRIEVAL COMPLETATO")
        print(f"   ⏱️  Tempo: {retrieval_time:.2f}s")
        print("="*70 + "\n")


# Nota: I parametri sono definiti come costanti all'inizio del file per facilità di modifica
# Questo script è completamente auto-contenuto e mostra esplicitamente ogni fase della pipeline RAG


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
        # Usa gemini-2.5-flash per valutazione DeepEval rigorosa
        eval_llm = get_langchain_model_provider(ModelProvider.GOOGLE, model_id="gemini-2.5-flash", temperature=0)
        eval_results = evaluate_rag(rag.chunks_query_retriever, llm=eval_llm, num_questions=3)

        # Mostra risultati valutazione DeepEval
        print(f"📊 Tipo valutazione: {eval_results.get('evaluation_type', 'N/A')}")
        print(f"🤖 Modello usato: {eval_results.get('model_used', 'N/A')}")
        print(f"❓ Domande valutate: {eval_results.get('questions_evaluated', 0)}")

        # Mostra punteggi medi
        if 'average_scores' in eval_results:
            avg = eval_results['average_scores']
            print("\n📈 PUNTEGGI MEDI (0-1, più alto = migliore):")
            print(f"• Correttezza: {avg.get('correctness', 0):.3f}")
            print(f"• Fedeltà: {avg.get('faithfulness', 0):.3f}")
            print(f"• Rilevanza: {avg.get('relevance', 0):.3f}")
        if 'results' in eval_results and eval_results['results']:
            print("\n📋 RISULTATI DETTAGLIATI:")
            for i, result in enumerate(eval_results['results'], 1):
                print(f"\n{i}. ❓ '{result.get('question', 'N/A')[:60]}...'")

                if result.get('scores'):
                    scores = result['scores']
                    print("   📊 Punteggi numerici:")
                    print(f"      • Correttezza: {scores.get('correctness', {}).get('score', 0):.3f}")
                    print(f"      • Fedeltà: {scores.get('faithfulness', {}).get('score', 0):.3f}")
                    print(f"      • Rilevanza: {scores.get('relevance', {}).get('score', 0):.3f}")
                    print("   📝 Valutazioni testuali:")
                    if 'correctness' in scores and scores['correctness'].get('reason'):
                        print(f"      • Correttezza: {scores['correctness']['reason'][:80]}...")
                    if 'faithfulness' in scores and scores['faithfulness'].get('reason'):
                        print(f"      • Fedeltà: {scores['faithfulness']['reason'][:80]}...")
                    if 'relevance' in scores and scores['relevance'].get('reason'):
                        print(f"      • Rilevanza: {scores['relevance']['reason'][:80]}...")
                elif 'error' in result:
                    print(f"   ❌ Errore: {result['error']}")

                if 'context_length' in result:
                    print(f"   📏 Contesto: {result['context_length']} caratteri")

        print(f"\n📝 Riepilogo: {eval_results.get('summary', 'Valutazione completata')}")


# 🎬 PUNTO DI INGRESSO: Avvio esecuzione script RAG
if __name__ == '__main__':
    # 🚀 ESECUZIONE RAG: Avvia elaborazione completa con parametri configurati
    main()


