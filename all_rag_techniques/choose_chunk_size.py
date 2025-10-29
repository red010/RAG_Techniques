"""
Sistema di Valutazione Dimensioni Chunk per RAG con Gemini

Questo script valuta diverse dimensioni di chunk per determinare la configurazione
ottimale del sistema RAG usando il testo "Sei personaggi in cerca d'autore" di Luigi Pirandello.

Il sistema misura:
- Tempo di risposta del retrieval
- Fedeltà (faithfulness) delle risposte
- Rilevanza (relevancy) dei documenti recuperati
"""

import random
import time
import os
import sys
from dotenv import load_dotenv

# Importazioni LangChain per caricamento e processamento documenti
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Importazioni LangChain per Google Gemini
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Vector store
from langchain_chroma import Chroma

# Carica variabili d'ambiente
load_dotenv()

# Imposta chiave API Google
os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')

# Aggiungi directory parent al path per modulo di valutazione
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from evaluation.evalute_rag import evaluate_rag


# Funzioni di utilità
def evaluate_response_time_and_accuracy(chunk_size, eval_questions, eval_documents, llm):
    """
    Valuta tempo di risposta medio, fedeltà e rilevanza delle risposte
    generate da Gemini per una data dimensione di chunk usando LangChain.

    Parametri:
    chunk_size (int): Dimensione dei chunk di testo processati.
    eval_questions (list): Lista delle domande di valutazione.
    eval_documents (list): Documenti per valutazione (oggetti Document di LangChain).
    llm (ChatGoogleGenerativeAI): Istanza LLM per generazione e valutazione.

    Ritorna:
    tuple: Tupla contenente (tempo_medio_risposta, fedeltà_media, rilevanza_media).
    """

    print(f"\n{'─'*80}")
    print(f"⚙️  VALUTAZIONE CHUNK SIZE: {chunk_size}")
    print(f"{'─'*80}")
    
    # Splitting del testo con LangChain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.1)  # 10% di sovrapposizione
    )
    chunks = text_splitter.split_documents(eval_documents)
    print(f"✂️  Creati {len(chunks)} chunks (overlap: 10%)")
    
    # Crea vector store con embeddings Gemini
    # Nota: Il modello embedding-001 supporta nativamente l'italiano
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(chunks, embeddings)
    
    # Crea retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Misura tempo di risposta medio
    print(f"\n⏱️  Misurazione tempo di risposta per {len(eval_questions)} domande...")
    total_time = 0
    for i, question in enumerate(eval_questions, 1):
        start_time = time.time()
        retriever.invoke(question)
        elapsed = time.time() - start_time
        total_time += elapsed
        print(f"   Domanda {i}/{len(eval_questions)}: {elapsed:.3f}s")
    
    avg_response_time = total_time / len(eval_questions)
    print(f"   ⏱️  Tempo medio: {avg_response_time:.3f}s")
    
    # Esegue valutazione con le domande hardcoded fornite
    print(f"\n📊 Esecuzione metriche di valutazione qualitativa...")
    eval_results = evaluate_rag(
        retriever=retriever,
        llm=llm,
        custom_questions=eval_questions  # Passa le domande hardcoded per la valutazione
    )
    
    # Estrai metriche
    avg_scores = eval_results.get('average_scores', {})
    avg_faithfulness = avg_scores.get('faithfulness', 0)
    avg_relevancy = avg_scores.get('relevance', 0)
    
    print(f"\n✅ Valutazione completata:")
    print(f"   ⏱️  Tempo medio risposta: {avg_response_time:.3f}s")
    print(f"   🎯 Fedeltà media: {avg_faithfulness:.3f}")
    print(f"   📍 Rilevanza media: {avg_relevancy:.3f}")
    
    return avg_response_time, avg_faithfulness, avg_relevancy


# Classe principale per il metodo RAG

class RAGEvaluator:
    """
    Valutatore RAG per testare diverse dimensioni di chunk.
    
    Carica il testo di Pirandello, genera domande di valutazione e
    misura le performance del sistema RAG per ogni chunk size.
    """
    
    def __init__(self, document_path, chunk_sizes):
        """
        Inizializza il valutatore RAG.
        
        Parametri:
        document_path (str): Percorso del file di testo da analizzare.
        chunk_sizes (list): Lista delle dimensioni di chunk da testare.
        """
        self.document_path = document_path
        self.chunk_sizes = chunk_sizes
        
        # Usa gemini-2.5-flash per valutazione (come in simple_rag_langchain_google.py)
        # Nota: Gemini 2.5 Flash supporta nativamente l'italiano
        self.llm_gemini = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )
        
        self.documents = self.load_documents()
        
        # Domande di valutazione hardcoded per "Sei personaggi in cerca d'autore"
        self.eval_questions = [
            "Quale commedia stavano provando gli Attori della Compagnia prima dell'arrivo dei Sei Personaggi e chi ne è l'autore?",
            "Come viene \"evocata\" Madama Pace sul palcoscenico e quale oggetto specifico, legato al suo commercio, ne innesca l'apparizione?",
            "Quali sono le versioni contrastanti fornite dal Padre, dalla Madre e dalla Figliastra riguardo all'allontanamento della Madre dalla casa coniugale?",
            "Spiega l'argomentazione del Padre sul perché i Personaggi sono \"più reali\" degli Attori. Qual è la differenza fondamentale tra la loro \"realtà\" e quella di un essere umano come il Capocomico?",
            "Analizzando il finale, dal colpo di rivoltella fino all'ultima indicazione di scena, in che modo il confine tra \"realtà\" e \"finzione\" viene definitivamente distrutto, lasciando il Capocomico e il pubblico nell'incertezza?"
        ]

    def load_documents(self):
        """
        Carica il documento di testo usando LangChain TextLoader.
        
        TextLoader supporta encoding UTF-8 per gestire correttamente
        i caratteri accentati italiani.
        """
        loader = TextLoader(
            self.document_path,
            encoding='utf-8'  # Necessario per caratteri accentati italiani
        )
        documents = loader.load()
        print(f"\n📄 Caricato documento: {os.path.basename(self.document_path)}")
        print(f"   📑 Numero pagine/sezioni: {len(documents)}")
        print(f"   📏 Lunghezza totale: {len(documents[0].page_content):,} caratteri")
        return documents


    def run(self):
        """
        Esegue la valutazione per tutte le dimensioni di chunk specificate.
        
        Per ogni chunk size, misura tempo di risposta, fedeltà e rilevanza.
        """
        print(f"\n{'='*80}")
        print(f"AVVIO VALUTAZIONE DIMENSIONI CHUNK RAG")
        print(f"{'='*80}")
        print(f"📄 Documento: {os.path.basename(self.document_path)}")
        print(f"📏 Chunk sizes da valutare: {self.chunk_sizes}")
        print(f"❓ Numero domande di valutazione: {len(self.eval_questions)}")
        print(f"\n{'─'*80}")
        print(f"DOMANDE DI VALUTAZIONE:")
        print(f"{'─'*80}")
        
        for i, q in enumerate(self.eval_questions, 1):
            # Word wrap per domande lunghe
            if len(q) > 76:
                words = q.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= 76:
                        current_line += word + " "
                    else:
                        lines.append(current_line.strip())
                        current_line = word + " "
                if current_line:
                    lines.append(current_line.strip())
                
                print(f"\n{i}. {lines[0]}")
                for line in lines[1:]:
                    print(f"   {line}")
            else:
                print(f"\n{i}. {q}")
        
        print(f"\n{'='*80}\n")
        
        # Struttura per memorizzare tutti i risultati
        all_results = []
        
        for chunk_size in self.chunk_sizes:
            avg_response_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy(
                chunk_size,
                self.eval_questions,
                self.documents,  # Usa tutti i documenti (in questo caso è un singolo file)
                self.llm_gemini
            )
            
            # Memorizza risultati
            all_results.append({
                'chunk_size': chunk_size,
                'time': avg_response_time,
                'faithfulness': avg_faithfulness,
                'relevancy': avg_relevancy
            })
        
        # Stampa riepilogo finale comparativo
        print(f"\n{'='*80}")
        print(f"📊 RIEPILOGO COMPARATIVO RISULTATI")
        print(f"{'='*80}\n")
        print(f"{'Chunk Size':<12} {'Tempo (s)':<12} {'Fedeltà':<12} {'Rilevanza':<12} {'Score Totale':<12}")
        print(f"{'─'*80}")
        
        for result in all_results:
            # Calcola score totale (media di fedeltà e rilevanza)
            total_score = (result['faithfulness'] + result['relevancy']) / 2
            print(f"{result['chunk_size']:<12} "
                  f"{result['time']:<12.3f} "
                  f"{result['faithfulness']:<12.3f} "
                  f"{result['relevancy']:<12.3f} "
                  f"{total_score:<12.3f}")
        
        # Identifica il migliore chunk size
        best_by_quality = max(all_results, key=lambda x: (x['faithfulness'] + x['relevancy']) / 2)
        best_by_speed = min(all_results, key=lambda x: x['time'])
        
        print(f"\n{'─'*80}")
        print(f"🏆 Migliore per qualità: Chunk size {best_by_quality['chunk_size']} "
              f"(score: {(best_by_quality['faithfulness'] + best_by_quality['relevancy']) / 2:.3f})")
        print(f"⚡ Più veloce: Chunk size {best_by_speed['chunk_size']} "
              f"(tempo: {best_by_speed['time']:.3f}s)")
        print(f"{'='*80}\n")


# Parsing argomenti da linea di comando

def parse_args():
    """
    Parsing degli argomenti CLI per configurare la valutazione.
    
    Default ottimizzati per il testo di Pirandello (118 KB):
    - Documento: sei_personaggi_in_cerca d_autore.txt
    - Chunk sizes: 256, 1024, 2048, 4096 (range completo per analisi comparativa)
    - Domande: 5 domande hardcoded specifiche sul testo di Pirandello
    """
    import argparse
    
    # Calcola percorso di default del documento di Pirandello
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    default_document = os.path.join(data_dir, 'sei_personaggi_in_cerca d_autore.txt')
    
    parser = argparse.ArgumentParser(
        description='Valutazione dimensioni chunk per sistema RAG con testo italiano',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi d'uso:
  # Valutazione standard con defaults (5 domande hardcoded su Pirandello)
  python choose_chunk_size.py
  
  # Test rapido con chunk sizes personalizzati
  python choose_chunk_size.py --chunk_sizes 512 1024
  
  # Documento custom (nota: le domande sono specifiche per Pirandello)
  python choose_chunk_size.py --document data/altro_testo.txt
        """
    )
    
    parser.add_argument('--document', type=str, default=default_document, 
                        help='Percorso del file di testo da analizzare '
                             '(default: sei_personaggi_in_cerca d_autore.txt)')
    parser.add_argument('--chunk_sizes', nargs='+', type=int, default=[256, 1024, 2048, 4096], 
                        help='Lista delle dimensioni di chunk da testare '
                             '(default: 256 1024 2048 4096)')
    return parser.parse_args()


if __name__ == "__main__":
    """
    Entry point dello script.
    
    Esegue la valutazione completa delle dimensioni chunk per il sistema RAG
    usando il testo di Pirandello, 5 domande hardcoded e Gemini API per embeddings.
    """
    args = parse_args()
    
    print("\n" + "="*80)
    print("🔬 SISTEMA DI VALUTAZIONE CHUNK SIZE PER RAG")
    print("="*80)
    print(f"📖 Testo: 'Sei personaggi in cerca d'autore' - Luigi Pirandello")
    print(f"🤖 Modello LLM: Google Gemini 2.5 Flash (supporto nativo italiano)")
    print(f"🧮 Modello Embeddings: Google embedding-001")
    print(f"❓ Domande: 5 domande hardcoded specifiche sul testo")
    print(f"📏 Chunk sizes da testare: {args.chunk_sizes}")
    print("="*80 + "\n")
    
    evaluator = RAGEvaluator(
        document_path=args.document, 
        chunk_sizes=args.chunk_sizes
    )
    evaluator.run()
