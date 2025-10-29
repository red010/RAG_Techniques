# %% [markdown]
# # RSE - Relevant Segment Extraction (Estrazione di Segmenti Rilevanti)
# 
# ## Panoramica
# 
# **TECNICA AVANZATA**: RSE (Relevant Segment Extraction) è un metodo per
# ricostruire segmenti contigui multi-chunk di testo a partire dai chunk recuperati.
# Questo passaggio avviene DOPO la ricerca vettoriale (e opzionalmente il reranking),
# ma PRIMA di presentare il contesto recuperato all'LLM. RSE assicura che chunk vicini
# siano presentati all'LLM nell'ordine in cui appaiono nel documento originale, e include
# anche chunk non marcati come rilevanti ma "intrappolati" tra chunk altamente rilevanti,
# migliorando ulteriormente il contesto fornito all'LLM.
# 
# ## Motivazione
# 
# Quando si suddividono documenti per RAG, scegliere la dimensione giusta dei chunk è
# un esercizio di gestione di trade-off:
# 
# - **Chunk grandi**: Forniscono migliore contesto all'LLM, ma rendono difficile
#   recuperare informazioni specifiche con precisione
# - **Chunk piccoli**: Permettono retrieval preciso, ma forniscono poco contesto
# 
# Alcune query (come domande fattuali semplici) si gestiscono meglio con chunk piccoli,
# mentre altre query (come domande di alto livello) richiedono chunk molto grandi.
# Ci sono query che possono essere risposte con una singola frase, e altre che
# richiedono interi paragrafi o capitoli per una risposta completa.
# 
# La maggior parte dei casi d'uso RAG reali deve affrontare una combinazione di
# questi tipi di query.
# 
# ## La Soluzione: RSE
# 
# Ciò di cui abbiamo veramente bisogno è un sistema più dinamico che possa:
# - Recuperare chunk brevi quando è tutto ciò che serve
# - Recuperare chunk molto grandi quando richiesto
# 
# **L'insight chiave**: I chunk rilevanti tendono a essere raggruppati in cluster
# all'interno dei loro documenti originali. RSE sfrutta questo fatto per ricostruire
# segmenti ottimali dinamicamente.
# 
# ## Componenti Chiave
# 
# ### 1. Chunking senza Overlap
# RSE richiede che i documenti siano suddivisi senza overlap. Questo permette di
# ricostruire sezioni del documento (segmenti) concatenando i chunk.
# 
# ### 2. Ottimizzazione RSE
# Dopo il retrieval e reranking standard, RSE:
# 1. Combina score di rilevanza assoluto e rank relativo
# 2. Sottrae una soglia costante (es. 0.2) per rendere i chunk irrilevanti negativi
# 3. Definisce il valore di un segmento come somma dei valori dei suoi chunk
# 4. Risolve una versione vincolata del "maximum subarray problem"
# 
# ### 3. Identificazione Cluster
# RSE identifica automaticamente cluster di chunk rilevanti consecutivi e li
# ricostruisce come segmenti contigui, includendo anche chunk intermedi non
# direttamente marcati come rilevanti.
# 
# ## Vantaggi
# 
# 1. **Contesto Completo**: Fornisce contesto più completo rispetto ai singoli chunk
# 2. **Flessibilità Dinamica**: Si adatta automaticamente alla complessità della query
# 3. **Recupero Implicito**: Include chunk rilevanti anche se non marcati dal reranker
# 4. **Prestazioni Superiori**: Miglioramento del 42.6% su benchmark KITE

# %% [markdown]
# 
# ![Relevant segment extraction](../images/relevant-segment-extraction.svg)
# 

# %% [markdown]
# # Setup e Imports

# %%
import os
import sys
import numpy as np
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from scipy.stats import beta as beta_dist

# Load environment variables
load_dotenv()
if not os.getenv('GEMINI_API_KEY'):
    raise ValueError("GEMINI_API_KEY non trovata nel file .env")
os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')

# %% [markdown]
# ### Configurazione Documento e Parametri

# %%
# Configurazione documento e modelli
script_dir = os.path.dirname(os.path.abspath(__file__))
default_data_dir = os.path.join(os.path.dirname(script_dir), 'data')
FILE_PATH = os.path.join(default_data_dir, "cambiamento_climatico.txt")

# Modelli Gemini
LANGUAGE_MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025"

# Parametri di chunking ottimizzati per RSE
CHUNK_SIZE = 250  # Chunk piccoli per massimizzare l'effetto RSE
CHUNK_OVERLAP = 0  # IMPORTANTE: RSE richiede overlap = 0

# Query che richiede risposta da multipli chunk contigui
QUERY = """Spiega in dettaglio il fenomeno dell'acidificazione degli oceani: 
quali sono le cause chimiche, gli effetti sugli organismi marini, 
e il legame con il cambiamento climatico."""

# %% [markdown]
# ### Funzioni Helper per Chunking e Reranking

# %%
def split_into_chunks(text: str, chunk_size: int) -> List[str]:
    """
    Suddivide il testo in chunk di dimensione specificata.
    
    IMPORTANTE PER RSE:
    - chunk_overlap DEVE essere 0
    - Questo permette di ricostruire il testo originale concatenando i chunk
    - Senza overlap, possiamo identificare precisamente i confini dei segmenti
    
    Args:
        text: Testo da suddividere
        chunk_size: Dimensione massima di ogni chunk
        
    Returns:
        Lista di chunk di testo
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=0,  # ZERO overlap è essenziale per RSE!
        length_function=len
    )
    texts = text_splitter.create_documents([text])
    chunks = [text.page_content for text in texts]
    return chunks

def transform(x: float) -> float:
    """
    Trasforma lo score di rilevanza per distribuirlo meglio tra 0 e 1.
    
    FUNZIONE BETA:
    Gli score di reranking tendono a concentrarsi vicino a 0 o 1.
    Questa funzione "stira" la distribuzione per renderla più uniforme,
    migliorando l'identificazione dei cluster.
    
    Args:
        x: Score di rilevanza (0-1)
        
    Returns:
        Score trasformato (0-1)
    """
    a, b = 0.4, 0.4  # Parametri della distribuzione beta
    return beta_dist.cdf(x, a, b)

def rerank_chunks_with_gemini(query: str, chunks: List[str]) -> Tuple[List[float], List[float]]:
    """
    Usa Gemini per calcolare lo score di rilevanza di ogni chunk rispetto alla query.
    
    RSE HA BISOGNO DI DUE METRICHE:
    1. similarity_scores: Score assoluto di rilevanza (0-1) per ogni chunk
    2. chunk_values: Combinazione di rank e similarity per l'ottimizzazione RSE
    
    IMPLEMENTAZIONE CON GEMINI:
    Per ogni chunk, chiediamo a Gemini di valutare la rilevanza su scala 0-10,
    poi normalizziamo a 0-1 e combiniamo con rank decay esponenziale.
    
    RANK DECAY:
    I chunk con rank migliore ottengono un bonus, simulando il comportamento
    di un reranker professionale come Cohere Rerank.
    
    Args:
        query: Query dell'utente
        chunks: Lista di chunk da valutare
        
    Returns:
        tuple: (similarity_scores, chunk_values)
    """
    llm = ChatGoogleGenerativeAI(
        model=LANGUAGE_MODEL_NAME,
        temperature=0
    )
    
    # Prompt per il reranking
    rerank_prompt = ChatPromptTemplate.from_messages([
        ("system", """Sei un esperto valutatore di rilevanza. 
Data una query e un chunk di testo, valuta quanto il chunk è rilevante per rispondere alla query.

Assegna uno score da 0 a 10:
- 0-2: Completamente irrilevante
- 3-4: Marginalmente rilevante
- 5-6: Parzialmente rilevante
- 7-8: Rilevante
- 9-10: Altamente rilevante e direttamente utile

Rispondi SOLO con il numero (es: "7")."""),
        ("human", """Query: {query}

Chunk: {chunk}

Score di rilevanza (0-10):""")
    ])
    
    print(f"\n🔄 Reranking {len(chunks)} chunk con Gemini...")
    print(f"   (Questo richiederà circa {len(chunks)} chiamate API)")
    
    # Valuta ogni chunk
    scores_raw = []
    for i, chunk in enumerate(chunks):
        if (i + 1) % 10 == 0:
            print(f"   Processati {i+1}/{len(chunks)} chunk...")
        
        try:
            response = llm.invoke(rerank_prompt.format_prompt(query=query, chunk=chunk).to_messages())
            score_text = response.content.strip()
            # Estrai il numero dalla risposta (gestisce "7" o "7/10" o "Score: 7")
            import re
            numbers = re.findall(r'\d+', score_text)
            if numbers:
                score = float(numbers[0]) / 10.0  # Normalizza 0-10 → 0-1
            else:
                score = 0.0
            scores_raw.append((i, score))
        except Exception as e:
            print(f"   ⚠️  Errore nel chunk {i}: {e}")
            scores_raw.append((i, 0.0))
    
    # Ordina per score decrescente (simula il reranking)
    scores_raw.sort(key=lambda x: x[1], reverse=True)
    
    # Calcola similarity_scores e chunk_values
    similarity_scores = [0.0] * len(chunks)
    chunk_values = [0.0] * len(chunks)
    decay_rate = 30  # Tasso di decadimento per il rank
    
    for rank, (chunk_idx, raw_score) in enumerate(scores_raw):
        # Applica trasformazione beta per distribuire meglio i valori
        transformed_score = transform(raw_score)
        similarity_scores[chunk_idx] = transformed_score
        
        # Combina rank e similarity con decay esponenziale
        # I chunk con rank migliore ottengono un boost
        chunk_values[chunk_idx] = np.exp(-rank/decay_rate) * transformed_score
    
    print(f"✅ Reranking completato")
    return similarity_scores, chunk_values

def print_relevance_distribution(chunk_values: List[float], 
                                  start_index: int = None, 
                                  end_index: int = None,
                                  top_n: int = 10) -> None:
    """
    Visualizza la distribuzione degli score di rilevanza in formato testuale.
    
    SCOPO DIDATTICO:
    Mostra come i chunk rilevanti tendono a formare cluster nel documento,
    che è l'insight fondamentale alla base di RSE.
    
    I cluster di chunk consecutivi con score alto indicano sezioni del documento
    particolarmente rilevanti per la query. RSE sfrutta questi pattern per
    ricostruire segmenti contigui ottimali.
    
    Args:
        chunk_values: Lista di score di rilevanza per ogni chunk
        start_index: Indice iniziale del range da visualizzare
        end_index: Indice finale del range da visualizzare
        top_n: Numero di chunk più rilevanti da mostrare
    """
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(chunk_values)
    
    print("\n" + "="*80)
    print("📊 DISTRIBUZIONE SCORE DI RILEVANZA")
    print("="*80)
    
    # Trova i chunk più rilevanti
    indexed_values = [(i, v) for i, v in enumerate(chunk_values)]
    top_chunks = sorted(indexed_values, key=lambda x: x[1], reverse=True)[:top_n]
    
    print(f"\n🏆 Top {top_n} chunk più rilevanti:")
    for rank, (idx, value) in enumerate(top_chunks, 1):
        bar = "█" * int(value * 50)  # Barra ASCII
        print(f"   {rank:2d}. Chunk {idx:3d}: {value:.3f} {bar}")
    
    # Identifica cluster di chunk consecutivi rilevanti
    print("\n🔍 Analisi Cluster (chunk consecutivi con score > 0.3):")
    in_cluster = False
    cluster_start = None
    clusters = []
    
    for i in range(start_index, end_index):
        if chunk_values[i] > 0.3:
            if not in_cluster:
                cluster_start = i
                in_cluster = True
        else:
            if in_cluster:
                clusters.append((cluster_start, i))
                in_cluster = False
    
    if in_cluster:
        clusters.append((cluster_start, end_index))
    
    if clusters:
        for cluster_idx, (start, end) in enumerate(clusters, 1):
            avg_score = np.mean(chunk_values[start:end])
            print(f"   Cluster {cluster_idx}: Chunk {start}-{end-1} (lunghezza: {end-start}, score medio: {avg_score:.3f})")
        print(f"\n💡 I {len(clusters)} cluster identificati dimostrano che i chunk rilevanti")
        print(f"   sono raggruppati, non sparsi casualmente nel documento!")
    else:
        print("   Nessun cluster significativo trovato")
    
    print("="*80)

def get_best_segments(relevance_values: list, max_length: int, overall_max_length: int, minimum_value: float) -> Tuple[List[Tuple[int, int]], List[float]]:
    """
    Trova i segmenti ottimali di chunk contigui usando ottimizzazione RSE.
    
    ALGORITMO RSE (CORE):
    Questo è il cuore della tecnica RSE. Risolve una versione vincolata del
    "maximum sum subarray problem" per identificare i migliori segmenti di
    chunk contigui.
    
    COME FUNZIONA:
    1. Cerca iterativamente il segmento con valore massimo
    2. Il valore di un segmento è la SOMMA dei valori dei suoi chunk
    3. Chunk con valore positivo = rilevanti, negativi = irrilevanti
    4. Include automaticamente chunk intermedi anche se poco rilevanti
    5. Rispetta vincoli di lunghezza massima
    
    VINCOLI:
    - max_length: Lunghezza massima di un singolo segmento (in # chunk)
    - overall_max_length: Lunghezza totale massima di tutti i segmenti
    - minimum_value: Valore minimo per considerare un segmento valido
    
    NOTA: Questa è un'implementazione semplificata per scopi didattici.
    Una versione production-ready è disponibile nella libreria dsRAG.
    
    Args:
        relevance_values: Lista di score di rilevanza (già con penalty sottratta)
        max_length: Lunghezza massima di un singolo segmento (# chunk)
        overall_max_length: Lunghezza massima totale (# chunk)
        minimum_value: Valore minimo per considerare un segmento
        
    Returns:
        tuple: (lista di segmenti come (start, end), lista di score)
    """
    best_segments = []
    scores = []
    total_length = 0
    
    while total_length < overall_max_length:
        # Trova il miglior segmento rimanente
        best_segment = None
        best_value = -1000
        
        for start in range(len(relevance_values)):
            # Salta punti di inizio con valore negativo
            if relevance_values[start] < 0:
                continue
            
            for end in range(start+1, min(start+max_length+1, len(relevance_values)+1)):
                # Salta punti di fine con valore negativo
                if relevance_values[end-1] < 0:
                    continue
                
                # Verifica se questo segmento si sovrappone con segmenti già selezionati
                if any(start < seg_end and end > seg_start for seg_start, seg_end in best_segments):
                    continue
                
                # Verifica se questo segmento supererebbe la lunghezza massima totale
                if total_length + end - start > overall_max_length:
                    continue
                
                # Calcola il valore del segmento come somma dei valori dei chunk
                segment_value = sum(relevance_values[start:end])
                if segment_value > best_value:
                    best_value = segment_value
                    best_segment = (start, end)
        
        # Se non troviamo un segmento valido, terminiamo
        if best_segment is None or best_value < minimum_value:
            break
        
        # Altrimenti, aggiungi il segmento alla lista
        best_segments.append(best_segment)
        scores.append(best_value)
        total_length += best_segment[1] - best_segment[0]
    
    return best_segments, scores

# %% [markdown]
# # FASE 1: Caricamento e Chunking Documento

# %%
print("\n" + "="*80)
print("📚 FASE 1: Caricamento e Chunking Documento")
print("="*80)

print(f"\n📋 Query di test:")
print(f"   {QUERY}")
print("\n💡 Questa query richiede informazioni da più parti del documento,")
print("   rendendo RSE particolarmente utile per ricostruire segmenti contigui.")

with open(FILE_PATH, 'r', encoding='utf-8') as file:
    text = file.read()

chunks = split_into_chunks(text, chunk_size=CHUNK_SIZE)
print(f"\n✅ Documento suddiviso in {len(chunks)} chunk")
print(f"   Dimensione chunk: {CHUNK_SIZE} caratteri")
print(f"   Overlap: {CHUNK_OVERLAP} caratteri (ZERO - essenziale per RSE!)")
print(f"\n💡 Chunk piccoli e overlap = 0 massimizzano l'effetto di RSE")

# %% [markdown]
# # FASE 2: Reranking dei Chunk con Gemini

# %%
print("\n" + "="*80)
print("🔍 FASE 2: Reranking dei Chunk")
print("="*80)

similarity_scores, chunk_values = rerank_chunks_with_gemini(QUERY, chunks)

# %% [markdown]
# # FASE 3: Analisi Distribuzione Score

# %%
# Visualizza distribuzione score
print_relevance_distribution(chunk_values, top_n=15)

# %% [markdown]
# # FASE 4: Ottimizzazione RSE - Ricerca Segmenti Ottimali

# %%
print("\n" + "="*80)
print("🎯 FASE 4: Ottimizzazione RSE - Ricerca Segmenti Ottimali")
print("="*80)

# Definisci parametri per l'ottimizzazione RSE
irrelevant_chunk_penalty = 0.2
max_length = 15  # Massimo numero di chunk per segmento
overall_max_length = 25  # Massimo numero totale di chunk
minimum_value = 0.5  # Valore minimo per considerare un segmento

print(f"\n⚙️  Parametri RSE:")
print(f"   Penalità chunk irrilevanti: {irrelevant_chunk_penalty}")
print(f"   Lunghezza massima segmento: {max_length} chunk")
print(f"   Lunghezza massima totale: {overall_max_length} chunk")
print(f"   Valore minimo segmento: {minimum_value}")

print(f"\n💡 La 'penalità chunk irrilevanti' sottrae {irrelevant_chunk_penalty} da ogni score,")
print(f"   rendendo negativi i chunk irrilevanti e positivi quelli rilevanti.")
print(f"   Questo permette di definire il valore di un segmento come somma semplice!")

# Sottrai penalty dai chunk values
relevance_values = [v - irrelevant_chunk_penalty for v in chunk_values]

# Esegui ottimizzazione
best_segments, scores = get_best_segments(
    relevance_values, 
    max_length, 
    overall_max_length, 
    minimum_value
)

print(f"\n✅ Trovati {len(best_segments)} segmenti ottimali:")
for i, (seg, score) in enumerate(zip(best_segments, scores), 1):
    start, end = seg
    print(f"\n   Segmento {i}:")
    print(f"      Chunk: {start}-{end-1} (lunghezza: {end-start})")
    print(f"      Score totale: {score:.3f}")
    print(f"      Score medio per chunk: {score/(end-start):.3f}")

# %% [markdown]
# # FASE 5: Visualizzazione Segmenti Recuperati

# %%
print("\n" + "="*80)
print("📖 FASE 5: Visualizzazione Segmenti Recuperati")
print("="*80)

for i, (seg, score) in enumerate(zip(best_segments, scores), 1):
    start, end = seg
    print(f"\n{'='*80}")
    print(f"SEGMENTO {i} (chunk {start}-{end-1}, score: {score:.3f})")
    print(f"{'='*80}")
    
    segment_text = " ".join([chunks[j] for j in range(start, end)])
    print(f"\n{segment_text}\n")
    print(f"{'-'*80}")
    
    # Analisi del segmento
    print(f"\n📊 Analisi Segmento {i}:")
    print(f"   Lunghezza: {len(segment_text)} caratteri ({end-start} chunk)")
    print(f"   Score medio chunk: {score/(end-start):.3f}")
    
    # Mostra quali chunk del segmento sono altamente rilevanti
    high_relevance_chunks = [j for j in range(start, end) if chunk_values[j] > 0.5]
    print(f"   Chunk ad alta rilevanza (>0.5): {len(high_relevance_chunks)}/{end-start}")
    
    if len(high_relevance_chunks) < (end-start):
        print(f"   💡 RSE ha incluso {(end-start) - len(high_relevance_chunks)} chunk intermedi")
        print(f"      non altamente rilevanti per completare il contesto!")

# %% [markdown]
# # Riepilogo Finale e Best Practices

# %%
print("\n" + "="*80)
print("✅ DEMO COMPLETATA: Relevant Segment Extraction")
print("="*80)

print("\n🎯 LEZIONI CHIAVE:")
print("   1. I chunk rilevanti tendono a formare CLUSTER nel documento originale")
print("      → RSE sfrutta questo pattern per ricostruire segmenti contigui")
print("   2. RSE include automaticamente chunk intermedi non marcati come rilevanti")
print("      → Fornisce contesto più completo rispetto al top-k retrieval")
print("   3. Il chunking SENZA overlap è essenziale per RSE")
print("      → Permette di ricostruire il testo originale concatenando chunk")
print("   4. RSE risolve dinamicamente il trade-off chunk piccoli vs grandi")
print("      → Recupera chunk brevi o segmenti lunghi in base alla query")

print("\n💡 QUANDO USARE RSE:")
print("   ✅ Query che richiedono risposte da più paragrafi")
print("   ✅ Documenti lunghi e strutturati (report, manuali, articoli)")
print("   ✅ Quando il contesto completo è cruciale per la risposta")
print("   ✅ In combinazione con reranking di qualità")

print("\n⚠️  QUANDO RSE HA MENO IMPATTO:")
print("   - Query che richiedono solo un singolo chunk")
print("   - Documenti molto brevi")
print("   - Quando la velocità è più importante della precisione")

print("\n⚙️  PARAMETRI CHIAVE DA OTTIMIZZARE:")
print("   • irrelevant_chunk_penalty (0.15-0.25):")
print("     Controlla quanto i chunk irrilevanti penalizzano un segmento")
print("   • max_length:")
print("     Limita la lunghezza di un singolo segmento")
print("   • overall_max_length:")
print("     Limita il contesto totale inviato all'LLM (costi)")
print("   • minimum_value:")
print("     Soglia minima per considerare un segmento valido")

print("\n📊 RISULTATI BENCHMARK (KITE):")
print("   RSE vs Top-k Retrieval (k=20):")
print("   • AI Papers:            4.5 → 7.9  (+75.6%)")
print("   • BVP Cloud 10-Ks:      2.6 → 4.4  (+69.2%)")
print("   • Sourcegraph Handbook: 5.7 → 6.6  (+15.8%)")
print("   • Supreme Court:        6.1 → 8.0  (+31.1%)")
print("   • Medio:                4.72 → 6.73 (+42.6%)")

print("\n🔬 BEST PRACTICES:")
print("   1. Usa chunk piccoli (200-400 caratteri) per massimizzare precision")
print("   2. IMPORTANTE: chunk_overlap DEVE essere 0 per RSE")
print("   3. Assicurati di avere un reranker di alta qualità")
print("   4. Combina RSE con Contextual Chunk Headers per risultati ottimali")
print("   5. Monitora la lunghezza dei segmenti per controllare costi LLM")
print("   6. Sperimenta con irrelevant_chunk_penalty per il tuo caso d'uso")

print("\n🎓 CONFRONTO CON ALTRE TECNICHE:")
print("   • Top-k Retrieval: Recupera i k chunk migliori indipendentemente")
print("     → RSE: Recupera segmenti contigui rispettando l'ordine originale")
print("   • Contextual Chunk Headers (CCH): Aggiunge contesto agli header")
print("     → RSE: Ricostruisce segmenti completi includendo chunk intermedi")
print("   • Combinazione RSE + CCH: Risultati ottimali su benchmark complessi")

print("\n💰 CONSIDERAZIONI SUI COSTI:")
print(f"   Questa demo ha richiesto ~{len(chunks)} chiamate API per il reranking")
print("   In produzione, considera:")
print("   • Usa un reranker dedicato (es. Google Vertex AI Ranking)")
print("   • Cachea i risultati di reranking per query frequenti")
print("   • Pre-calcola gli score per documenti statici")

print("\n" + "="*80 + "\n")

# %% [markdown]
# ## 📚 Approfondimenti Tecnici
# 
# ### Il Maximum Subarray Problem
# 
# RSE risolve una versione vincolata del classico problema algoritmico del
# "maximum subarray sum". A differenza dell'algoritmo di Kadane che ha complessità
# O(n), RSE usa un approccio brute-force con euristiche perché deve rispettare
# vincoli aggiuntivi:
# 
# 1. **Lunghezza massima del segmento**: Limita i chunk in un singolo segmento
# 2. **Lunghezza totale**: Limita il contesto totale inviato all'LLM
# 3. **Segmenti non sovrapposti**: Ogni chunk può apparire in un solo segmento
# 4. **Valore minimo**: Ignora segmenti con score troppo basso
# 
# ### Perché Sottrarre una Penalty?
# 
# La sottrazione della penalty (`irrelevant_chunk_penalty`) è cruciale perché:
# 
# - Trasforma chunk irrilevanti in valori negativi
# - Permette di definire il valore del segmento come somma semplice
# - I chunk irrilevanti riducono il valore del segmento
# - Ma se sono "intrappolati" tra chunk rilevanti, possono comunque essere inclusi
# 
# Esempio:
# ```
# Chunk values: [0.8, 0.9, 0.3, 0.7, 0.2]
# Dopo penalty (0.2): [0.6, 0.7, 0.1, 0.5, 0.0]
# Segmento [0-3]: valore = 0.6+0.7+0.1+0.5 = 1.9
# Il chunk 2 (0.3) è incluso anche se poco rilevante!
# ```
# 
# ### Implementazione Production-Ready
# 
# Per un'implementazione ottimizzata di RSE, vedi la libreria
# [dsRAG](https://github.com/D-Star-AI/dsRAG) che include:
# 
# - Algoritmi ottimizzati per performance
# - Gestione di documenti multipli
# - Integrazione con vector store
# - Caching intelligente dei risultati
# - Metriche di valutazione

# %% [markdown]
# ![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--relevant-segment-extraction)
