"""
Semantic Chunking RAG - Confronto tra Chunking Semantico e Simple Chunking

Questo script dimostra l'uso del Semantic Chunking, una tecnica avanzata di suddivisione
del testo che considera la similarità semantica tra frasi per determinare i confini dei chunk,
anziché utilizzare una dimensione fissa.

Il Semantic Chunking:
- Analizza le somiglianze semantiche tra frasi consecutive usando embeddings
- Identifica "punti di rottura" naturali dove il significato cambia
- Crea chunk di dimensioni variabili basati sul contenuto semantico
- Può preservare meglio il contesto e la coerenza tematica

Confronto con Simple Chunking:
- Simple Chunking: divide il testo in chunk di dimensione fissa con overlap
- Semantic Chunking: divide il testo dove cambia il significato
- Trade-off: Semantic chunking è più lento ma potenzialmente più accurato

Utilizzo:
    # Modalità standard (solo semantic chunking)
    python semantic_chunking.py --path ../data/document.pdf

    # Modalità esperimento (confronto semantic vs simple)
    python semantic_chunking.py --path ../data/document.pdf --experiment
"""

import time
import os
import sys
import argparse
from dotenv import load_dotenv

# Aggiungi la directory parent al path per lavorare con i notebook
# Deve essere fatto PRIMA di importare helper_functions
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Ora possiamo importare da helper_functions e altri moduli
from helper_functions import read_pdf_to_string
from langchain_experimental.text_splitter import SemanticChunker, BreakpointThresholdType
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Carica variabili d'ambiente dal file .env (API key di Gemini)
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')


# ============================================================================
# CONFIGURAZIONE PARAMETRI DI DEFAULT
# ============================================================================
# 
# IMPORTANTE: Tutti i valori di default sono centralizzati qui per facilità di modifica.
# Modificare questi valori per cambiare il comportamento di default dello script
# senza dover cercare nel codice.
#
# NOTA: Questi valori sono ottimizzati per il documento "cambiamento_climatico.txt"
#       e possono richiedere aggiustamenti per altri documenti.
#

# Documento da analizzare
DEFAULT_DOCUMENT_NAME = 'cambiamento_climatico.txt'

# Parametri retrieval
DEFAULT_N_RETRIEVED = 3  # Numero di chunk da recuperare per ogni query

# Parametri Semantic Chunking
# 'gradient' è molto sensibile, ideale per testi densi e complessi
DEFAULT_BREAKPOINT_TYPE = "gradient"  # Opzioni: percentile, standard_deviation, interquartile, gradient
DEFAULT_BREAKPOINT_AMOUNT = 60  # Valore soglia per breakpoint (interpretazione dipende dal tipo)

# Parametri Simple Chunking
# Valori ottimizzati per testi italiani densi: chunk piccoli con overlap ~25%
DEFAULT_CHUNK_SIZE = 512  # Dimensione chunk in caratteri
DEFAULT_CHUNK_OVERLAP = 64  # Sovrapposizione tra chunk (circa 25% di chunk_size)

# Query di test di default
# Query complessa che richiede comprensione di concetti correlati ma distinti
DEFAULT_QUERY = "Qual è la differenza tra la causa diretta dello sbiancamento dei coralli e il meccanismo chimico che porta all'acidificazione degli oceani?"

# Modello embeddings
DEFAULT_EMBEDDING_MODEL = "models/embedding-001"  # Gemini embedding model (supporto nativo italiano)


# ============================================================================
# CLASSE SEMANTIC CHUNKING RAG
# ============================================================================

class SemanticChunkingRAG:
    """
    Classe per gestire il processo RAG con Semantic Chunking.
    
    Il Semantic Chunking divide il testo in chunk basandosi sulla similarità semantica
    tra frasi consecutive. Quando la similarità scende sotto una certa soglia (breakpoint),
    viene creato un nuovo chunk.
    
    Tipi di Breakpoint Threshold:
    ================================
    - 'percentile': Usa un percentile della distribuzione delle distanze.
      Es: 90 significa che solo il 10% delle distanze più grandi causerà una rottura.
      → Più conservativo, chunk più grandi.
    
    - 'standard_deviation': Usa deviazioni standard dalla media.
      Es: 2.0 significa che distanze > (media + 2*std) causano rotture.
      → Bilanciato, sensibile a outliers.
    
    - 'interquartile': Usa il range interquartile (Q1-Q3).
      → Robusto agli outliers, chunk di dimensione media.
    
    - 'gradient': Identifica picchi nel gradiente delle distanze.
      → Molto sensibile, identifica cambiamenti sottili di argomento.
      → IDEALE per testi complessi e densi come saggi scientifici o analisi approfondite.
    
    Best Practice:
    - Per testi tecnici/strutturati: 'percentile' con valore alto (85-95)
    - Per narrativa/contenuto fluido: 'standard_deviation' con valore medio (1.5-2.0)
    - Per documenti eterogenei o saggi complessi: 'gradient' per massima granularità semantica
    - Per testi in italiano: il modello Gemini embedding-001 supporta nativamente l'italiano
    """

    def __init__(self, path, n_retrieved=2, embeddings=None, breakpoint_type: BreakpointThresholdType = "percentile",
                 breakpoint_amount=90):
        """
        Inizializza SemanticChunkingRAG processando il documento con chunking semantico.

        Args:
            path (str): Percorso al file PDF da processare.
            n_retrieved (int): Numero di chunk da recuperare per ogni query (default: 2).
                              Più alto = più contesto ma potenzialmente più rumore.
            embeddings: Modello di embedding da usare. Se None, usa Gemini embedding-001.
            breakpoint_type (str): Tipo di soglia per identificare i breakpoint semantici.
                                  Opzioni: 'percentile', 'standard_deviation', 'interquartile', 'gradient'
            breakpoint_amount (float): Valore della soglia (interpretazione dipende dal tipo).
                                      Es: 90 per percentile = 90° percentile
        """
        print("\n" + "─"*80)
        print("🔬 INIZIALIZZAZIONE SEMANTIC CHUNKING RAG")
        print("─"*80)
        
        # Legge il contenuto del documento (PDF o TXT)
        print(f"📖 Caricamento documento: {os.path.basename(path)}")
        if path.lower().endswith('.txt'):
            # Carica file di testo direttamente
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"   ✓ Caricato file TXT")
        else:
            # Carica PDF usando helper function
            content = read_pdf_to_string(path)
            print(f"   ✓ Caricato file PDF")
        print(f"   ✓ {len(content):,} caratteri totali")

        # Usa embeddings forniti o inizializza embeddings Gemini
        # Il modello embedding-001 di Gemini supporta 768 dimensioni
        self.embeddings = embeddings if embeddings else GoogleGenerativeAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
        print(f"   ✓ Modello embeddings: Gemini {DEFAULT_EMBEDDING_MODEL}")

        # Inizializza il semantic chunker con i parametri specificati
        # Il chunker analizza il testo frase per frase, calcola embeddings,
        # e identifica dove la similarità semantica scende sotto la soglia
        self.semantic_chunker = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=breakpoint_amount
        )
        print(f"   ✓ Breakpoint: {breakpoint_type} ({breakpoint_amount})")

        # Esegue il semantic chunking e misura il tempo
        # Nota: Questo processo richiede embed di ogni frase, quindi può essere lento
        print(f"\n✂️  Esecuzione semantic chunking...")
        start_time = time.time()
        self.semantic_docs = self.semantic_chunker.create_documents([content])
        self.time_records = {'Chunking': time.time() - start_time}
        print(f"   ✓ Completato in {self.time_records['Chunking']:.2f} secondi")
        print(f"   ✓ Creati {len(self.semantic_docs)} chunks semantici")
        
        # Calcola statistiche sui chunk creati
        chunk_sizes = [len(doc.page_content) for doc in self.semantic_docs]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)
        print(f"   📊 Dimensioni chunk: min={min_size}, avg={int(avg_size)}, max={max_size}")

        # Crea vector store Chroma e retriever dai chunk semantici
        # Chroma è un vector database leggero e veloce, ottimo per prototipi
        # IMPORTANTE: Specifica un nome di collection unico per evitare conflitti
        print(f"\n💾 Creazione vector store Chroma e indicizzazione...")
        start_vectorstore = time.time()
        self.semantic_vectorstore = Chroma.from_documents(
            self.semantic_docs, 
            self.embeddings,
            collection_name="semantic_chunks"  # Nome collection unico
        )
        print(f"   ✓ Vector store creato in {time.time() - start_vectorstore:.2f} secondi")
        
        # Il retriever usa ricerca per similarità coseno per trovare i k chunk più rilevanti
        self.semantic_retriever = self.semantic_vectorstore.as_retriever(search_kwargs={"k": n_retrieved})
        print(f"   ✓ Retriever configurato (k={n_retrieved})")

    def run(self, query):
        """
        Esegue una query sul sistema RAG e visualizza i risultati.
        
        Il processo:
        1. La query viene convertita in embedding usando lo stesso modello
        2. Chroma trova i k chunk più simili usando similarità coseno
        3. I chunk vengono recuperati e visualizzati
        
        Args:
            query (str): La query in linguaggio naturale da processare.

        Returns:
            dict: Dizionario con metriche temporali:
                  - 'Chunking': Tempo per creare i chunk (già calcolato in __init__)
                  - 'Retrieval': Tempo per recuperare chunk rilevanti per la query
        """
        print(f"\n🔍 Query: '{query}'")
        print("─"*80)
        
        # Misura il tempo per il retrieval semantico
        # Questo include: embedding della query + ricerca vettoriale in Chroma
        start_time = time.time()
        semantic_context = self.semantic_retriever.invoke(query)
        self.time_records['Retrieval'] = time.time() - start_time
        print(f"⏱️  Tempo Retrieval: {self.time_records['Retrieval']:.3f} secondi")

        # Visualizza i chunk recuperati completamente
        print(f"\n📄 Recuperati {len(semantic_context)} chunks rilevanti:\n")
        for i, doc in enumerate(semantic_context, 1):
            print(f"{'='*80}")
            print(f"📄 CHUNK {i} | Lunghezza: {len(doc.page_content)} caratteri")
            print(f"{'='*80}")
            print(doc.page_content)
            print(f"{'='*80}\n")

        return self.time_records


# ============================================================================
# CLASSE SIMPLE CHUNKING RAG (PER CONFRONTO)
# ============================================================================

class SimpleChunkingRAG:
    """
    Classe per gestire il chunking semplice (fixed-size) per confronto con semantic chunking.
    
    Il Simple Chunking (o Fixed-Size Chunking) è l'approccio tradizionale:
    - Divide il testo in chunk di dimensione fissa (es. 1000 caratteri)
    - Usa un overlap tra chunk consecutivi per preservare continuità
    - Molto veloce ma può spezzare concetti in punti arbitrari
    - Non considera il significato o la struttura del testo
    
    Vantaggi:
    + Velocissimo (no embedding necessari per chunking)
    + Dimensione chunk prevedibile
    + Facile da implementare e debuggare
    
    Svantaggi:
    - Può spezzare frasi, paragrafi o concetti a metà
    - L'overlap è arbitrario, non basato sul contenuto
    - Chunk possono contenere più argomenti non correlati
    
    Quando usarlo:
    - Quando la velocità è critica
    - Per testi molto omogenei
    - Come baseline per confronti
    """
    
    def __init__(self, path, n_retrieved=2, embeddings=None, chunk_size=1000, chunk_overlap=200):
        """
        Inizializza SimpleChunkingRAG usando chunking a dimensione fissa.
        
        Args:
            path (str): Percorso al file PDF da processare.
            n_retrieved (int): Numero di chunk da recuperare per query (default: 2).
            embeddings: Modello di embedding da usare per il vector store.
            chunk_size (int): Dimensione target di ogni chunk in caratteri (default: 1000).
                             RecursiveCharacterTextSplitter cerca di rispettarla ma può variare.
            chunk_overlap (int): Sovrapposizione tra chunk consecutivi in caratteri (default: 200).
                                Aiuta a mantenere continuità tra chunk adiacenti.
        """
        print("\n" + "─"*80)
        print("📏 INIZIALIZZAZIONE SIMPLE CHUNKING RAG")
        print("─"*80)
        
        # Legge il contenuto del documento (PDF o TXT)
        print(f"📖 Caricamento documento: {os.path.basename(path)}")
        if path.lower().endswith('.txt'):
            # Carica file di testo direttamente
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"   ✓ Caricato file TXT")
        else:
            # Carica PDF usando helper function
            content = read_pdf_to_string(path)
            print(f"   ✓ Caricato file PDF")
        print(f"   ✓ {len(content):,} caratteri totali")
        
        # Usa embeddings forniti o inizializza Gemini embeddings
        # Nota: Gli embeddings sono usati solo per il vector store, NON per il chunking
        self.embeddings = embeddings if embeddings else GoogleGenerativeAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
        print(f"   ✓ Modello embeddings: Gemini {DEFAULT_EMBEDDING_MODEL}")
        
        # Inizializza lo splitter a dimensione fissa
        # RecursiveCharacterTextSplitter prova a dividere su separatori naturali (\n\n, \n, spazi)
        # ma rispetta sempre chunk_size come limite superiore
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        print(f"   ✓ Configurazione: chunk_size={chunk_size}, overlap={chunk_overlap}")
        
        # Esegue il chunking semplice e misura il tempo
        # Questo è molto più veloce del semantic chunking (no embeddings necessari)
        print(f"\n✂️  Esecuzione simple chunking...")
        start_time = time.time()
        self.simple_docs = text_splitter.create_documents([content])
        self.time_records = {'Chunking': time.time() - start_time}
        print(f"   ✓ Completato in {self.time_records['Chunking']:.2f} secondi")
        print(f"   ✓ Creati {len(self.simple_docs)} chunks")
        
        # Calcola statistiche sui chunk creati
        chunk_sizes = [len(doc.page_content) for doc in self.simple_docs]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)
        print(f"   📊 Dimensioni chunk: min={min_size}, avg={int(avg_size)}, max={max_size}")
        
        # Crea vector store e retriever dai chunk semplici
        # Usa lo stesso sistema (Chroma + Gemini embeddings) per fair comparison
        # IMPORTANTE: Specifica un nome di collection unico per evitare conflitti
        print(f"\n💾 Creazione vector store Chroma e indicizzazione...")
        start_vectorstore = time.time()
        self.simple_vectorstore = Chroma.from_documents(
            self.simple_docs, 
            self.embeddings,
            collection_name="simple_chunks"  # Nome collection unico
        )
        print(f"   ✓ Vector store creato in {time.time() - start_vectorstore:.2f} secondi")
        
        self.simple_retriever = self.simple_vectorstore.as_retriever(search_kwargs={"k": n_retrieved})
        print(f"   ✓ Retriever configurato (k={n_retrieved})")
    
    def run(self, query):
        """
        Esegue una query sul sistema RAG con simple chunking e visualizza i risultati.
        
        Args:
            query (str): La query in linguaggio naturale da processare.
        
        Returns:
            dict: Dizionario con metriche temporali (Chunking, Retrieval).
        """
        print(f"\n🔍 Query: '{query}'")
        print("─"*80)
        
        # Misura il tempo per il retrieval
        start_time = time.time()
        simple_context = self.simple_retriever.invoke(query)
        self.time_records['Retrieval'] = time.time() - start_time
        print(f"⏱️  Tempo Retrieval: {self.time_records['Retrieval']:.3f} secondi")
        
        # Visualizza i chunk recuperati completamente
        print(f"\n📄 Recuperati {len(simple_context)} chunks rilevanti:\n")
        for i, doc in enumerate(simple_context, 1):
            print(f"{'='*80}")
            print(f"📄 CHUNK {i} | Lunghezza: {len(doc.page_content)} caratteri")
            print(f"{'='*80}")
            print(doc.page_content)
            print(f"{'='*80}\n")
        
        return self.time_records


# ============================================================================
# PARSING ARGOMENTI DA LINEA DI COMANDO
# ============================================================================

def parse_args():
    """
    Definisce e parsea gli argomenti da linea di comando.
    
    Argomenti principali:
    - --path: Percorso al documento PDF da processare
    - --experiment: Attiva modalità confronto tra semantic e simple chunking
    - --breakpoint_threshold_type: Strategia per identificare breakpoint semantici
    - --breakpoint_threshold_amount: Valore soglia per i breakpoint
    
    Returns:
        argparse.Namespace: Oggetto con tutti gli argomenti parsati
    """
    # Calcola path assoluto al documento di default usando la costante
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_doc_path = os.path.join(os.path.dirname(script_dir), 'data', DEFAULT_DOCUMENT_NAME)
    
    parser = argparse.ArgumentParser(
        description="Processa un documento con Semantic Chunking RAG e Gemini API. "
                    "Supporta modalità esperimento per confrontare semantic vs simple chunking. "
                    "Ottimizzato per testi in italiano.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:
  # Modalità standard con semantic chunking
  python semantic_chunking.py --path ../data/documento.txt
  
  # Modalità esperimento: confronto semantic vs simple
  python semantic_chunking.py --experiment
  
  # Personalizza breakpoint threshold
  python semantic_chunking.py --breakpoint_threshold_type gradient --breakpoint_threshold_amount 90
  
  # Personalizza query di test
  python semantic_chunking.py --query "Quali sono le cause principali?"
        """
    )
    
    parser.add_argument(
        "--path", 
        type=str, 
        default=default_doc_path,
        help=f"Percorso al file di testo da processare (default: {DEFAULT_DOCUMENT_NAME}). "
             f"Supporta file .txt e .pdf."
    )
    
    parser.add_argument(
        "--n_retrieved", 
        type=int, 
        default=DEFAULT_N_RETRIEVED,
        help=f"Numero di chunk da recuperare per ogni query (default: {DEFAULT_N_RETRIEVED}). "
             "Aumentare per più contesto, diminuire per più precisione."
    )
    
    parser.add_argument(
        "--breakpoint_threshold_type", 
        type=str,
        choices=["percentile", "standard_deviation", "interquartile", "gradient"],
        default=DEFAULT_BREAKPOINT_TYPE,
        help=f"Tipo di soglia per identificare breakpoint semantici (default: {DEFAULT_BREAKPOINT_TYPE}). "
             "Opzioni: 'percentile' (conservativo), 'standard_deviation' (bilanciato), "
             "'interquartile' (robusto), 'gradient' (molto sensibile, ideale per testi complessi)."
    )
    
    parser.add_argument(
        "--breakpoint_threshold_amount", 
        type=float, 
        default=DEFAULT_BREAKPOINT_AMOUNT,
        help=f"Valore della soglia breakpoint (default: {DEFAULT_BREAKPOINT_AMOUNT}). "
             "Per 'gradient': 90 identifica picchi significativi nel gradiente delle distanze. "
             "Per 'percentile': 90 = 90° percentile. "
             "Per 'standard_deviation': 2.0 = 2 deviazioni standard. "
             "Valori più alti = chunk più grandi e coesi."
    )
    
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=DEFAULT_CHUNK_SIZE,
        help=f"Dimensione di ogni chunk per simple chunking in caratteri (default: {DEFAULT_CHUNK_SIZE}). "
             "Usato solo in modalità --experiment. Valori bassi (200-300) per testi densi."
    )
    
    parser.add_argument(
        "--chunk_overlap", 
        type=int, 
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Sovrapposizione tra chunk consecutivi per simple chunking in caratteri (default: {DEFAULT_CHUNK_OVERLAP}). "
             "Usato solo in modalità --experiment. Circa 25%% di chunk_size per massima continuità."
    )
    
    parser.add_argument(
        "--query", 
        type=str, 
        default=DEFAULT_QUERY,
        help="Query di test per il retriever (default: domanda complessa su coralli e acidificazione). "
             "Usa una domanda rilevante al contenuto del documento in italiano."
    )
    
    parser.add_argument(
        "--experiment", 
        action="store_true",
        help="Attiva modalità esperimento per confrontare performance tra semantic chunking "
             "e simple chunking sulla stessa query. Mostra metriche comparative dettagliate."
    )

    return parser.parse_args()


# ============================================================================
# FUNZIONE PRINCIPALE
# ============================================================================

def main(args):
    """
    Funzione principale per eseguire semantic chunking e opzionalmente confrontarlo con simple chunking.
    
    Due modalità di esecuzione:
    
    1. Modalità Standard (default):
       - Esegue solo semantic chunking
       - Mostra risultati per la query specificata
       - Utile per uso normale del sistema RAG
    
    2. Modalità Esperimento (--experiment):
       - Esegue sia semantic che simple chunking
       - Confronta le performance su metriche chiave
       - Evidenzia differenze in velocità e qualità
       - Utile per capire quando usare semantic chunking
    
    Metriche confrontate in modalità esperimento:
    - Numero di chunk creati: Indica granularità
    - Tempo chunking: Overhead del semantic chunking
    - Tempo retrieval: Dipende da numero chunk e dimensione vector store
    - Tempo totale: Overhead complessivo
    - Qualità chunk recuperati: Valutazione manuale necessaria
    
    Args:
        args (argparse.Namespace): Argomenti parsati da linea di comando
    """
    
    if args.experiment:
        # ====================================================================
        # MODALITÀ ESPERIMENTO: Confronto Semantic vs Simple Chunking
        # ====================================================================
        
        print("\n" + "="*80)
        print("🔬 MODALITÀ ESPERIMENTO: Confronto Semantic vs Simple Chunking")
        print("="*80)
        print(f"\n📖 Documento: {os.path.basename(args.path)}")
        print(f"🔍 Query di test: '{args.query}'")
        print(f"📊 Chunk recuperati per metodo: {args.n_retrieved}")
        print("\n" + "="*80)
        
        # Crea embeddings condivisi per fairness nel confronto
        # Entrambi i metodi useranno lo stesso modello per vector store
        print("\n⚙️  Inizializzazione embeddings condivisi...")
        shared_embeddings = GoogleGenerativeAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
        print(f"   ✓ Gemini {DEFAULT_EMBEDDING_MODEL} caricato")
        
        # ----------------------------------------------------------------
        # FASE 1: Semantic Chunking
        # ----------------------------------------------------------------
        print("\n" + "="*80)
        print("📊 FASE 1/2: SEMANTIC CHUNKING")
        print("="*80)
        semantic_rag = SemanticChunkingRAG(
            path=args.path,
            n_retrieved=args.n_retrieved,
            embeddings=shared_embeddings,
            breakpoint_type=args.breakpoint_threshold_type,
            breakpoint_amount=args.breakpoint_threshold_amount
        )
        
        # ----------------------------------------------------------------
        # FASE 2: Simple Chunking
        # ----------------------------------------------------------------
        print("\n" + "="*80)
        print("📊 FASE 2/2: SIMPLE CHUNKING")
        print("="*80)
        simple_rag = SimpleChunkingRAG(
            path=args.path,
            n_retrieved=args.n_retrieved,
            embeddings=shared_embeddings,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        # ----------------------------------------------------------------
        # FASE 3: Esecuzione Query e Confronto Risultati
        # ----------------------------------------------------------------
        print("\n" + "="*80)
        print("🔍 FASE 3: ESECUZIONE QUERY E CONFRONTO")
        print("="*80)
        
        print("\n" + "─"*80)
        print("📊 RISULTATI SEMANTIC CHUNKING:")
        print("─"*80)
        semantic_times = semantic_rag.run(args.query)
        
        print("\n" + "─"*80)
        print("📊 RISULTATI SIMPLE CHUNKING:")
        print("─"*80)
        simple_times = simple_rag.run(args.query)
        
        # ----------------------------------------------------------------
        # FASE 4: Confronto Performance Finale
        # ----------------------------------------------------------------
        print("\n" + "="*80)
        print("📊 CONFRONTO PERFORMANCE FINALE")
        print("="*80)
        
        # Calcola metriche comparative
        chunk_diff = len(semantic_rag.semantic_docs) - len(simple_rag.simple_docs)
        chunk_diff_pct = (chunk_diff / len(simple_rag.simple_docs)) * 100
        
        chunking_diff = semantic_times['Chunking'] - simple_times['Chunking']
        chunking_diff_pct = (chunking_diff / simple_times['Chunking']) * 100
        
        retrieval_diff = semantic_times['Retrieval'] - simple_times['Retrieval']
        retrieval_diff_pct = (retrieval_diff / simple_times['Retrieval']) * 100 if simple_times['Retrieval'] > 0 else 0
        
        total_semantic = sum(semantic_times.values())
        total_simple = sum(simple_times.values())
        total_diff = total_semantic - total_simple
        total_diff_pct = (total_diff / total_simple) * 100
        
        # Tabella comparativa
        print(f"\n{'Metrica':<30} {'Semantic':<15} {'Simple':<15} {'Differenza':<20}")
        print("─"*80)
        print(f"{'Chunks creati':<30} {len(semantic_rag.semantic_docs):<15} "
              f"{len(simple_rag.simple_docs):<15} {chunk_diff:>+6} ({chunk_diff_pct:>+6.1f}%)")
        print(f"{'Tempo chunking (s)':<30} {semantic_times['Chunking']:<15.3f} "
              f"{simple_times['Chunking']:<15.3f} {chunking_diff:>+6.3f} ({chunking_diff_pct:>+6.1f}%)")
        print(f"{'Tempo retrieval (s)':<30} {semantic_times['Retrieval']:<15.3f} "
              f"{simple_times['Retrieval']:<15.3f} {retrieval_diff:>+6.3f} ({retrieval_diff_pct:>+6.1f}%)")
        print(f"{'Tempo totale (s)':<30} {total_semantic:<15.3f} "
              f"{total_simple:<15.3f} {total_diff:>+6.3f} ({total_diff_pct:>+6.1f}%)")
        
        # Interpretazione risultati
        print("\n" + "─"*80)
        print("💡 INTERPRETAZIONE RISULTATI:")
        print("─"*80)
        
        if chunk_diff < 0:
            print(f"✓ Semantic chunking ha creato {abs(chunk_diff)} chunk in MENO")
            print("  → Chunk più grandi, potenzialmente più contesto per chunk")
        else:
            print(f"✓ Semantic chunking ha creato {chunk_diff} chunk in PIÙ")
            print("  → Chunk più piccoli, potenzialmente più granulari")
        
        if chunking_diff > 0:
            print(f"\n⚠ Semantic chunking è {chunking_diff:.2f}s più LENTO nel chunking")
            print(f"  → Overhead: {chunking_diff_pct:.1f}% (dovuto al calcolo di embeddings)")
        
        if total_diff > 0:
            print(f"\n📊 Tempo totale semantic: {total_diff:.2f}s in più ({total_diff_pct:.1f}%)")
        else:
            print(f"\n📊 Tempo totale semantic: {abs(total_diff):.2f}s in meno ({abs(total_diff_pct):.1f}%)")
        
        print("\n💭 Valuta manualmente la QUALITÀ dei chunk recuperati sopra!")
        print("   I chunk semantici mantengono meglio coerenza tematica?")
        print("   I chunk semplici hanno rotture arbitrarie in mezzo a concetti?")
        
        print("\n" + "="*80)
        print("🎓 RACCOMANDAZIONI:")
        print("="*80)
        print("Usa SEMANTIC CHUNKING quando:")
        print("  • La qualità è prioritaria rispetto alla velocità")
        print("  • Il documento ha struttura tematica chiara")
        print("  • Devi preservare coerenza semantica nei chunk")
        print("\nUsa SIMPLE CHUNKING quando:")
        print("  • La velocità è critica (batch processing)")
        print("  • Il documento è molto omogeneo")
        print("  • Hai vincoli di risorse (memoria, API calls)")
        print("="*80 + "\n")
        
    else:
        # ====================================================================
        # MODALITÀ STANDARD: Solo Semantic Chunking
        # ====================================================================
        
        print("\n" + "="*80)
        print("🔬 SEMANTIC CHUNKING RAG - Modalità Standard")
        print("="*80)
        print(f"\n📖 Documento: {os.path.basename(args.path)}")
        print(f"🔍 Query di test: '{args.query}'")
        print(f"⚙️  Breakpoint: {args.breakpoint_threshold_type} ({args.breakpoint_threshold_amount})")
        print(f"📊 Chunk da recuperare: {args.n_retrieved}")
        print("\n" + "="*80)
        
        # Inizializza e esegui semantic chunking
        semantic_rag = SemanticChunkingRAG(
            path=args.path,
            n_retrieved=args.n_retrieved,
            breakpoint_type=args.breakpoint_threshold_type,
            breakpoint_amount=args.breakpoint_threshold_amount
        )
        
        # Esegui query
        print("\n" + "="*80)
        print("🔍 ESECUZIONE QUERY")
        print("="*80)
        times = semantic_rag.run(args.query)
        
        # Riepilogo finale
        print("\n" + "="*80)
        print("📊 RIEPILOGO PERFORMANCE")
        print("="*80)
        print(f"⏱️  Tempo chunking:  {times['Chunking']:.3f}s")
        print(f"⏱️  Tempo retrieval: {times['Retrieval']:.3f}s")
        print(f"⏱️  Tempo totale:    {sum(times.values()):.3f}s")
        print(f"✂️  Chunks creati:   {len(semantic_rag.semantic_docs)}")
        print(f"📄 Chunks recuperati: {args.n_retrieved}")
        print("="*80 + "\n")
        
        print("💡 Suggerimento: Usa --experiment per confrontare con simple chunking!")
        print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Parsea argomenti e esegui funzione principale
    main(parse_args())
