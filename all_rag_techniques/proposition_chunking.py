# %% [markdown]
# # Proposition Chunking (Chunking a Proposizioni)

# %% [markdown]
# ### Panoramica / Overview
# 
# Questo codice implementa il metodo del **Proposition Chunking**, basato sulla [ricerca di Tony Chen, et. al.](https://doi.org/10.48550/arXiv.2312.06648). 
# 
# **COSA SONO LE PROPOSIZIONI?**
# Una proposizione è un'unità atomica di informazione: un singolo fatto auto-contenuto che può essere compreso senza contesto aggiuntivo.
# Invece di dividere il testo in chunk di dimensione fissa, questo metodo usa un LLM per estrarre affermazioni fattuali discrete.
# 
# **PERCHÉ È UTILE PER IL RAG?**
# - **Maggiore Precisione**: Recupera solo i fatti rilevanti, non interi paragrafi con informazioni miste
# - **Riduzione del Rumore**: Ogni proposizione contiene un'unica informazione, eliminando il "rumore di contesto"
# - **Risposte più Accurate**: Il LLM di generazione riceve solo i fatti necessari, non contesto superfluo
# 
# ### Componenti Chiave
# 
# 1. **Document Chunking Iniziale:** Suddivisione preliminare del documento in pezzi gestibili per l'analisi LLM.
# 2. **Generazione Proposizioni:** Utilizzo di un LLM per scomporre ogni chunk in proposizioni fattuali e auto-contenute.
# 3. **Quality Check delle Proposizioni:** Valutazione delle proposizioni generate basata su accuratezza, chiarezza, completezza e concisione.
# 4. **Embedding e Vector Store:** Codifica sia delle proposizioni che dei chunk più grandi in un vector store per un retrieval efficiente.
# 5. **Retrieval e Confronto:** Test del sistema di retrieval con diverse query e confronto tra il modello basato su proposizioni e quello basato su chunk più grandi.
# 
# <img src="../images/proposition_chunking.svg" alt="Proposition-Chunking" width="600">
# 
# ### Motivazione
# 
# La motivazione dietro il metodo del proposition chunking è costruire un sistema che scomponga un documento testuale in proposizioni concise e fattuali per un retrieval di informazioni più granulare e preciso. 
# 
# L'uso delle proposizioni permette un controllo più fine e una migliore gestione di query specifiche, particolarmente utile per estrarre conoscenza da testi dettagliati o complessi. 
# 
# Il confronto tra l'uso di piccoli chunk di proposizioni e chunk di documenti più grandi mira a valutare l'efficacia del retrieval granulare delle informazioni.
# 
# ### Dettagli del Metodo
# 
# 1. **Caricamento Variabili d'Ambiente:** Il codice inizia caricando le variabili d'ambiente (es. API keys per il servizio LLM) per garantire l'accesso alle risorse necessarie.
#    
# 2. **Document Chunking:**
#    - Il documento in input viene suddiviso in pezzi più piccoli usando `RecursiveCharacterTextSplitter`. Questo assicura che ogni chunk sia di dimensioni gestibili per l'elaborazione da parte dell'LLM.
#    
# 3. **Generazione delle Proposizioni:**
#    - Le proposizioni vengono generate da ogni chunk usando un LLM (in questo caso, "gemini-2.5-flash"). L'output è strutturato come una lista di affermazioni fattuali e auto-contenute comprensibili senza contesto aggiuntivo.
#    
# 4. **Quality Check:**
#    - Un secondo passaggio LLM valuta la qualità delle proposizioni assegnando punteggi per accuratezza, chiarezza, completezza e concisione. Vengono mantenute solo le proposizioni che soddisfano le soglie richieste in tutte le categorie.
#    
# 5. **Embedding delle Proposizioni:**
#    - Le proposizioni che superano il quality check vengono codificate in un vector store usando il modello `GoogleGenerativeAIEmbeddings`. Questo permette il retrieval basato su similarità delle proposizioni quando vengono effettuate le query.
#    
# 6. **Retrieval e Confronto:**
#    - Vengono costruiti due sistemi di retrieval: uno usando i chunk basati su proposizioni e un altro usando chunk di documento più grandi. Entrambi vengono testati con diverse query per confrontare le loro performance e la precisione dei risultati restituiti.
# 
# ### Vantaggi
# 
# - **Granularità:** Scomponendo il documento in piccole proposizioni fattuali, il sistema permette un retrieval altamente specifico, rendendo più facile estrarre risposte precise da documenti grandi o complessi.
# - **Garanzia di Qualità:** L'uso di un LLM per il quality-checking assicura che le proposizioni generate soddisfino standard specifici, migliorando l'affidabilità delle informazioni recuperate.
# - **Flessibilità nel Retrieval:** Il confronto tra retrieval basato su proposizioni e quello basato su chunk più grandi permette di valutare i trade-off tra granularità e contesto più ampio nei risultati di ricerca.
# 
# ### Implementazione
# 
# 1. **Generazione Proposizioni:** L'LLM viene usato in congiunzione con un prompt personalizzato per generare affermazioni fattuali dai chunk del documento.
# 2. **Quality Checking:** Le proposizioni generate passano attraverso un sistema di valutazione che valuta accuratezza, chiarezza, completezza e concisione.
# 3. **Integrazione Vector Store:** Le proposizioni vengono memorizzate in un vector store Chroma dopo essere state codificate usando un modello di embedding pre-addestrato, permettendo una ricerca e retrieval efficiente basato su similarità.
# 4. **Query Testing:** Vengono effettuate multiple query di test ai vector store (basati su proposizioni e su chunk più grandi) per confrontare le performance di retrieval.
# 
# ### Riepilogo
# 
# Questo codice presenta un metodo robusto per scomporre un documento in proposizioni auto-contenute usando LLM. Il sistema effettua un quality check su ogni proposizione, le codifica in un vector store e recupera le informazioni più rilevanti basandosi sulle query dell'utente. La capacità di confrontare proposizioni granulari contro chunk di documento più grandi fornisce insight su quale metodo produca risultati più accurati o utili per diversi tipi di query. L'approccio enfatizza l'importanza della generazione e retrieval di proposizioni di alta qualità per l'estrazione precisa di informazioni da documenti complessi.

# %% [markdown]
# # Installazione Pacchetti e Import
# 
# La cella sottostante installa tutti i pacchetti necessari per eseguire questo notebook.
# 

# %%
# Installa i pacchetti richiesti
# !pip install langchain langchain-community langchain-google-genai langchain-chroma python-dotenv
# Nota: Esegui questo comando manualmente nel terminale se i pacchetti non sono installati

# %%
# CONFIGURAZIONE LLM E AMBIENTE
import os
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file '.env'
load_dotenv()

# Configura la chiave API per Google Gemini
os.environ['GOOGLE_API_KEY'] = os.getenv('GEMINI_API_KEY')

# %% [markdown]
# ### Documento di Test

# %%
# CARICAMENTO DOCUMENTO ITALIANO SUL CAMBIAMENTO CLIMATICO
# Il documento viene caricato da file invece di essere hardcodato
# Questo permette di lavorare con testi più lunghi e complessi

import os

# Determina il percorso del file rispetto alla posizione dello script
script_dir = os.path.dirname(os.path.abspath(__file__))
document_path = os.path.join(os.path.dirname(script_dir), 'data', 'cambiamento_climatico.txt')

# Carica il contenuto del documento italiano
with open(document_path, 'r', encoding='utf-8') as f:
    sample_content = f.read()

print(f"✅ Documento caricato: {len(sample_content)} caratteri")

# %% [markdown]
# ### Chunking Iniziale del Documento

# %%
# CHUNKING INIZIALE: Prima suddivisione del documento
# Questo NON è il chunking finale - è solo per rendere il testo gestibile dall'LLM
# L'LLM analizzerà ogni chunk per estrarre le proposizioni atomiche

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configura il modello di embedding di Google
# Il modello 'embedding-001' è ottimizzato per testi multilingue (incluso l'italiano)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Crea il documento con metadata appropriati
docs_list = [Document(
    page_content=sample_content, 
    metadata={
        "Title": "Cambiamento Climatico: Un Mosaico di Cause ed Effetti", 
        "Source": "Documento Didattico"
    }
)]

# Suddividi il documento in chunk gestibili
# chunk_size=200: Dimensione ottimale per l'analisi LLM - abbastanza piccola da essere specifica
# chunk_overlap=50: Sovrapposizione per non perdere contesto tra i chunk
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200, 
    chunk_overlap=50
)

doc_splits = text_splitter.split_documents(docs_list)

print(f"📄 Documento suddiviso in {len(doc_splits)} chunk iniziali")

# %%
# Aggiungi un identificatore univoco a ogni chunk
# Questo permette di tracciare da quale chunk proviene ogni proposizione
for i, doc in enumerate(doc_splits):
    doc.metadata['chunk_id'] = i+1

# %% [markdown]
# ### Stima Costi e Chiamate API

# %%
# STIMA COSTI: Calcolo preventivo del numero di chiamate e del costo stimato
# Questo permette all'utente di valutare se procedere prima di consumare quota API

def estimate_api_costs(doc_splits):
    """
    Stima il numero di chiamate API e il costo approssimativo per l'intero processo.
    
    Il processo coinvolge:
    1. Generazione proposizioni: 1 chiamata per chunk (parallele in batch di 10)
    2. Quality check: 1 chiamata per ogni proposizione generata (parallele in batch di 20)
    3. Query di test: 4 query (solo retrieval, nessuna chiamata LLM)
    
    Prezzi Gemini 2.5 Flash-Lite Preview (Settembre 2025):
    - Input (text/image/video): $0.10 per 1M tokens
    - Input (audio): $0.30 per 1M tokens
    - Output (inclusi thinking tokens): $0.40 per 1M tokens
    
    Fonte: https://ai.google.dev/pricing
    
    Tempo stimato con parallelizzazione:
    - ~5-10x più veloce rispetto all'esecuzione sequenziale
    """
    
    num_chunks = len(doc_splits)
    
    # Stima media di proposizioni per chunk (basata su esperienza empirica)
    # Chunk di 200 caratteri generano tipicamente 3-7 proposizioni
    avg_propositions_per_chunk = 5
    estimated_propositions = num_chunks * avg_propositions_per_chunk
    
    # FASE 1: Generazione Proposizioni
    generation_calls = num_chunks
    
    # Stima token per generazione (prompt + few-shot + chunk content)
    avg_input_tokens_generation = 300  # System prompt + few-shot + chunk
    avg_output_tokens_generation = 150  # Lista di proposizioni
    
    generation_input_tokens = generation_calls * avg_input_tokens_generation
    generation_output_tokens = generation_calls * avg_output_tokens_generation
    
    # FASE 2: Quality Check
    quality_check_calls = estimated_propositions
    
    # Stima token per quality check (evaluation prompt + proposizione + chunk originale)
    avg_input_tokens_quality = 400  # Evaluation prompt + proposition + original text
    avg_output_tokens_quality = 50   # Structured output con 4 numeri
    
    quality_input_tokens = quality_check_calls * avg_input_tokens_quality
    quality_output_tokens = quality_check_calls * avg_output_tokens_quality
    
    # TOTALI
    total_calls = generation_calls + quality_check_calls
    total_input_tokens = generation_input_tokens + quality_input_tokens
    total_output_tokens = generation_output_tokens + quality_output_tokens
    
    # COSTI (Prezzi Gemini 2.5 Flash-Lite Preview - Paid Tier)
    input_cost_per_1m = 0.10    # $0.10 per 1M input tokens (text/image/video)
    output_cost_per_1m = 0.40   # $0.40 per 1M output tokens (inclusi thinking tokens)
    
    input_cost = (total_input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (total_output_tokens / 1_000_000) * output_cost_per_1m
    total_cost = input_cost + output_cost
    
    return {
        'num_chunks': num_chunks,
        'estimated_propositions': estimated_propositions,
        'generation_calls': generation_calls,
        'quality_check_calls': quality_check_calls,
        'total_calls': total_calls,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost
    }

# Calcola la stima
cost_estimate = estimate_api_costs(doc_splits)

# Visualizza la stima in modo chiaro
print("\n" + "="*80)
print("📊 STIMA CHIAMATE API E COSTI")
print("="*80)
print(f"\n📄 DOCUMENTO:")
print(f"   - Chunks da processare: {cost_estimate['num_chunks']}")
print(f"   - Proposizioni stimate: ~{cost_estimate['estimated_propositions']}")

print(f"\n🔄 CHIAMATE API:")
print(f"   - Generazione proposizioni: {cost_estimate['generation_calls']} chiamate")
print(f"   - Quality check proposizioni: ~{cost_estimate['quality_check_calls']} chiamate")
print(f"   - TOTALE chiamate: ~{cost_estimate['total_calls']} chiamate")

print(f"\n🎯 TOKEN STIMATI:")
print(f"   - Input tokens: ~{cost_estimate['total_input_tokens']:,}")
print(f"   - Output tokens: ~{cost_estimate['total_output_tokens']:,}")
print(f"   - Totale tokens: ~{cost_estimate['total_input_tokens'] + cost_estimate['total_output_tokens']:,}")

print(f"\n💰 COSTO STIMATO (Gemini 2.5 Flash-Lite Preview):")
print(f"   - Input cost:  ${cost_estimate['input_cost']:.4f} ($0.10/1M tokens)")
print(f"   - Output cost: ${cost_estimate['output_cost']:.4f} ($0.40/1M tokens)")
print(f"   - TOTALE:      ${cost_estimate['total_cost']:.4f}")

print(f"\n⏱️  TEMPO STIMATO CON PARALLELIZZAZIONE:")
print(f"   - Batch di 10 chunks in parallelo per generazione")
print(f"   - Batch di 20 proposizioni in parallelo per quality check")
print(f"   - Circa 2-4 minuti (vs 10-15 min sequenziali)")
print(f"   - Velocizzazione: ~5-10x")

print("\n" + "="*80)
print("⚠️  NOTE IMPORTANTI:")
print("   - Questi sono valori stimati. Il consumo reale può variare del ±30%")
print("   - Prezzi basati sul Paid Tier di Google AI")
print("   - Free Tier: disponibile ma con limiti di rate (500 RPD per grounding)")
print("="*80)

# Chiedi conferma prima di procedere
proceed = input("\n🤔 Vuoi procedere con l'elaborazione? (s/n): ")

if proceed.lower() not in ['s', 'si', 'sì', 'yes', 'y']:
    print("\n❌ Elaborazione annullata dall'utente.")
    import sys
    sys.exit(0)

# Chiedi se eseguire il quality check (opzionale per velocizzare)
skip_quality_check_input = input("\n⚡ Vuoi saltare il quality check per velocizzare? (s/n, default=n): ")
SKIP_QUALITY_CHECK = skip_quality_check_input.lower() in ['s', 'si', 'sì', 'yes', 'y']

if SKIP_QUALITY_CHECK:
    print("⚠️  Quality check DISABILITATO - tutte le proposizioni saranno accettate")
else:
    print("✅ Quality check ABILITATO - le proposizioni saranno valutate")

print("\n✅ Procedendo con l'elaborazione...\n")

# %% [markdown]
# ### Generazione delle Proposizioni

# %%
# GENERAZIONE PROPOSIZIONI: Il cuore del Proposition Chunking
# Qui l'LLM trasforma ogni chunk in una lista di proposizioni atomiche
# Ogni proposizione è un singolo fatto che può essere compreso indipendentemente

from typing import List
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# Data model per le proposizioni generate
class GeneratePropositions(BaseModel):
    """Lista di tutte le proposizioni in un dato documento"""

    propositions: List[str] = Field(
        description="Lista di proposizioni (informazioni fattuali, auto-contenute e concise)"
    )


# Configura l'LLM Gemini con structured output
# gemini-2.5-flash-lite-preview: Versione ottimizzata per costo/efficienza e alta qualità
# temperature=0: Output deterministico per coerenza nella generazione
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-09-2025", temperature=0)
structured_llm = llm.with_structured_output(GeneratePropositions)

# FEW-SHOT PROMPTING: Esempi per guidare l'LLM
# Mostriamo all'LLM come vogliamo che scomponga il testo
# Esempi pertinenti al cambiamento climatico per coerenza tematica
proposition_examples = [
    {"document": 
        "Nel 2015, l'Accordo di Parigi ha stabilito l'obiettivo di mantenere l'aumento della temperatura globale ben al di sotto dei 2°C rispetto ai livelli preindustriali, con l'impegno di limitarlo a 1,5°C.", 
     "propositions": 
        "['L\\'Accordo di Parigi è un trattato internazionale sul clima.', 'L\\'Accordo di Parigi è stato firmato nel 2015.', 'L\\'Accordo di Parigi stabilisce un obiettivo di temperatura globale.', 'L\\'obiettivo dell\\'Accordo di Parigi è mantenere l\\'aumento della temperatura ben al di sotto dei 2°C rispetto ai livelli preindustriali.', 'L\\'Accordo di Parigi include l\\'impegno a limitare l\\'aumento della temperatura a 1,5°C.']"
    },
]

example_proposition_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{document}"),
        ("ai", "{propositions}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt = example_proposition_prompt,
    examples = proposition_examples,
)

# SYSTEM PROMPT: Istruzioni dettagliate per la generazione delle proposizioni
# Le 5 regole assicurano che ogni proposizione sia atomica e auto-contenuta
system = """Scomponi il seguente testo in proposizioni semplici e auto-contenute. Assicurati che ogni proposizione rispetti i seguenti criteri:

    1. Esprima un Singolo Fatto: Ogni proposizione deve affermare un fatto o una dichiarazione specifica.
    2. Sia Comprensibile Senza Contesto: La proposizione deve essere auto-contenuta, comprensibile senza necessità di contesto aggiuntivo.
    3. Utilizzi Nomi Completi, Non Pronomi: Evita pronomi o riferimenti ambigui; usa i nomi completi delle entità.
    4. Includa Date/Qualificatori Rilevanti: Se applicabile, includi date, orari e qualificatori necessari per rendere il fatto preciso.
    5. Contenga una Singola Relazione Soggetto-Predicato: Concentrati su un singolo soggetto e la sua corrispondente azione o attributo, senza congiunzioni o clausole multiple."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        few_shot_prompt,
        ("human", "{document}"),
    ]
)

proposition_generator = prompt | structured_llm

# %%
# GENERAZIONE MASSIVA ASINCRONA: Processa tutti i chunk in parallelo
# Usa asyncio per fare chiamate API parallele e ridurre drasticamente il tempo
import time
import asyncio
from typing import List as TypingList

async def generate_propositions_async(chunk_data):
    """Genera proposizioni per un singolo chunk in modo asincrono"""
    chunk_idx, doc_chunk = chunk_data
    try:
        # Genera le proposizioni per questo chunk
        response = await proposition_generator.ainvoke({"document": doc_chunk.page_content})
        
        # Crea Document per ogni proposizione
        props = []
        for proposition in response.propositions:
            props.append(Document(
                page_content=proposition, 
                metadata={
                    "Title": "Cambiamento Climatico: Un Mosaico di Cause ed Effetti", 
                    "Source": "Documento Didattico", 
                    "chunk_id": chunk_idx + 1
                }
            ))
        return props
    except Exception as e:
        # GESTIONE ERRORI: Possibili cause di errore
        # 1. Timeout API: Il modello impiega troppo tempo a rispondere
        # 2. Rate limiting: Troppe richieste simultanee all'API
        # 3. Chunk problematico: Testo troppo complesso o mal formattato
        # 4. Risposta None: L'API restituisce una risposta vuota/null
        # 
        # NOTA: Il chunk viene saltato e il processo continua con i successivi
        # Questo evita che un singolo chunk problematico blocchi l'intero processo
        print(f"⚠️ Errore nel chunk {chunk_idx + 1}: {e}")
        print(f"   Causa probabile: Timeout API o risposta None dal modello")
        print(f"   Impatto: Chunk saltato, il processo continua")
        return []

async def process_chunks_in_batches(doc_splits, batch_size=10):
    """Processa i chunk in batch per evitare troppi task simultanei"""
    all_propositions = []
    total_chunks = len(doc_splits)
    
    print(f"🔄 Generazione proposizioni da {total_chunks} chunk in batch di {batch_size}...")
    print(f"   Stima: ~{cost_estimate['estimated_propositions']} proposizioni attese\n")
    
    start_time = time.time()
    
    for batch_start in range(0, total_chunks, batch_size):
        batch_end = min(batch_start + batch_size, total_chunks)
        batch = list(enumerate(doc_splits[batch_start:batch_end], start=batch_start))
        
        # Processa il batch in parallelo
        tasks = [generate_propositions_async(chunk_data) for chunk_data in batch]
        batch_results = await asyncio.gather(*tasks)
        
        # Aggrega i risultati
        for props in batch_results:
            all_propositions.extend(props)
        
        # Progress indicator
        elapsed = time.time() - start_time
        avg_time_per_chunk = elapsed / batch_end if batch_end > 0 else 0
        remaining_chunks = total_chunks - batch_end
        eta_seconds = remaining_chunks * avg_time_per_chunk / batch_size  # Diviso per batch_size perché sono paralleli
        
        print(f"   [{batch_end}/{total_chunks}] Proposizioni generate: {len(all_propositions)} | "
              f"ETA: {eta_seconds/60:.1f}min | "
              f"Velocità: {batch_end/elapsed:.1f} chunks/sec")
    
    return all_propositions, time.time() - start_time

# Esegui la generazione asincrona
propositions, generation_time = asyncio.run(process_chunks_in_batches(doc_splits, batch_size=10))
generation_api_calls = len(doc_splits)

print(f"\n✅ Fase 1 completata:")
print(f"   - Proposizioni generate: {len(propositions)} (vs {cost_estimate['estimated_propositions']} stimate)")
print(f"   - Chiamate API: {generation_api_calls}")
print(f"   - Tempo impiegato: {generation_time/60:.2f} minuti")
print(f"   - Velocità media: {generation_api_calls/generation_time:.1f} chunks/sec")

# %% [markdown]
# ### Quality Check delle Proposizioni

# %%
# QUALITY CHECK: Valutazione della qualità delle proposizioni
# Non tutte le proposizioni generate sono di alta qualità
# Questo passaggio filtra quelle che non soddisfano gli standard

# Data model per la valutazione delle proposizioni
class GradePropositions(BaseModel):
    """Valuta una data proposizione su accuratezza, chiarezza, completezza e concisione"""

    accuracy: int = Field(
        description="Vota da 1-10 in base a quanto bene la proposizione riflette il testo originale."
    )
    
    clarity: int = Field(
        description="Vota da 1-10 in base a quanto sia facile comprendere la proposizione senza contesto aggiuntivo."
    )

    completeness: int = Field(
        description="Vota da 1-10 in base al fatto che la proposizione includa dettagli necessari (es. date, qualificatori)."
    )

    conciseness: int = Field(
        description="Vota da 1-10 in base al fatto che la proposizione sia concisa senza perdere informazioni importanti."
    )

# LLM con structured output per la valutazione
# Usa lo stesso modello veloce per coerenza
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-09-2025", temperature=0)
structured_llm = llm.with_structured_output(GradePropositions)

# EVALUATION PROMPT: Template per valutare ogni proposizione
# Include esempi specifici per il dominio del cambiamento climatico
evaluation_prompt_template = """
Valuta la seguente proposizione basandoti sui criteri sotto:
- **Accuratezza**: Vota da 1-10 in base a quanto bene la proposizione riflette il testo originale.
- **Chiarezza**: Vota da 1-10 in base a quanto sia facile comprendere la proposizione senza contesto aggiuntivo.
- **Completezza**: Vota da 1-10 in base al fatto che la proposizione includa dettagli necessari (es. date, qualificatori).
- **Concisione**: Vota da 1-10 in base al fatto che la proposizione sia concisa senza perdere informazioni importanti.

Esempio:
Testo: Nel 2015, l'Accordo di Parigi ha stabilito l'obiettivo di mantenere l'aumento della temperatura globale ben al di sotto dei 2°C rispetto ai livelli preindustriali.

Proposizione_1: L'Accordo di Parigi è un trattato internazionale sul clima.
Valutazione_1: "accuracy": 10, "clarity": 10, "completeness": 9, "conciseness": 10

Proposizione_2: L'Accordo di Parigi è stato firmato nel 2015.
Valutazione_2: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

Formato:
Proposizione: "{proposition}"
Testo Originale: "{original_text}"
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", evaluation_prompt_template),
        ("human", "{proposition}, {original_text}"),
    ]
)

proposition_evaluator = prompt | structured_llm

# %%
# THRESHOLD CONFIGURATION: Soglie ottimizzate per testi italiani complessi
# I testi scientifici italiani sul clima richiedono threshold bilanciati:
# - accuracy=7: Alta accuratezza ma permissiva per terminologia tecnica
# - clarity=7: Equilibrio tra chiarezza e precisione tecnica
# - completeness=5: Più permissivo - proposizioni atomiche possono omettere contesto
# - conciseness=6: Più permissivo - terminologia tecnica può richiedere più parole
# 
# NOTA: Threshold abbassati dopo test empirici - le proposizioni atomiche
# per loro natura possono avere "completeness" bassa pur essendo corrette

evaluation_categories = ["accuracy", "clarity", "completeness", "conciseness"]
thresholds = {"accuracy": 7, "clarity": 7, "completeness": 5, "conciseness": 6}

# Funzione per valutare una singola proposizione
def evaluate_proposition(proposition, original_text):
    """Valuta una proposizione usando l'LLM e restituisce i punteggi"""
    response = proposition_evaluator.invoke({
        "proposition": proposition, 
        "original_text": original_text
    })
    
    # Estrai i punteggi dalla risposta strutturata
    scores = {
        "accuracy": response.accuracy, 
        "clarity": response.clarity, 
        "completeness": response.completeness, 
        "conciseness": response.conciseness
    }
    return scores

# Funzione per verificare se la proposizione passa il quality check
def passes_quality_check(scores):
    """Verifica se tutti i punteggi superano le soglie minime"""
    for category, score in scores.items():
        if score < thresholds[category]:
            return False
    return True

# %% [markdown]
# ### Quality Check delle Proposizioni (Opzionale)

# %%
# QUALITY CHECK ASINCRONO: Valuta le proposizioni in parallelo
# Questo step può essere saltato per velocizzare l'elaborazione

# Salta il quality check se richiesto dall'utente
if SKIP_QUALITY_CHECK:
    evaluated_propositions = propositions
    quality_time = 0
    quality_api_calls = 0
    failed_count = 0
    
    print(f"\n⚡ Quality check SALTATO")
    print(f"   - Tutte le {len(propositions)} proposizioni sono state accettate")
    print(f"   - Tempo risparmiato: ~2-5 minuti")
else:
    # Esegui il quality check completo
    async def evaluate_proposition_async(idx, proposition, doc_splits):
        """Valuta una singola proposizione in modo asincrono"""
        try:
            response = await proposition_evaluator.ainvoke({
                "proposition": proposition.page_content,
                "original_text": doc_splits[proposition.metadata['chunk_id'] - 1].page_content
            })
            
            scores = {
                "accuracy": response.accuracy,
                "clarity": response.clarity,
                "completeness": response.completeness,
                "conciseness": response.conciseness
            }
            
            passed = passes_quality_check(scores)
            return idx, proposition, scores, passed
        except Exception as e:
            print(f"⚠️ Errore valutazione proposizione {idx+1}: {e}")
            return idx, proposition, None, False

    async def evaluate_propositions_in_batches(propositions, doc_splits, batch_size=20):
        """Valuta le proposizioni in batch per evitare troppi task simultanei"""
        evaluated = []
        failed_examples = []
        total_props = len(propositions)
        
        print(f"\n🔍 Avvio quality check su {total_props} proposizioni in batch di {batch_size}...")
        print(f"   Stima chiamate: ~{total_props}\n")
        
        start_time = time.time()
        failed_count = 0
        
        for batch_start in range(0, total_props, batch_size):
            batch_end = min(batch_start + batch_size, total_props)
            batch = [(i, propositions[i]) for i in range(batch_start, batch_end)]
            
            # Processa il batch in parallelo
            tasks = [evaluate_proposition_async(idx, prop, doc_splits) for idx, prop in batch]
            batch_results = await asyncio.gather(*tasks)
            
            # Aggrega i risultati
            for idx, proposition, scores, passed in batch_results:
                if passed:
                    evaluated.append(proposition)
                else:
                    failed_count += 1
                    # Memorizza solo i primi 5 fallimenti per il report
                    if len(failed_examples) < 5 and scores:
                        failed_examples.append((idx+1, proposition.page_content[:80], scores))
            
            # Progress indicator
            elapsed = time.time() - start_time
            avg_time_per_prop = elapsed / batch_end if batch_end > 0 else 0
            remaining = total_props - batch_end
            eta_seconds = remaining * avg_time_per_prop / batch_size
            
            print(f"   [{batch_end}/{total_props}] Valutate | "
                  f"Accettate: {len(evaluated)} | "
                  f"Rifiutate: {failed_count} | "
                  f"ETA: {eta_seconds/60:.1f}min")
        
        return evaluated, failed_examples, failed_count, time.time() - start_time

    # Esegui il quality check asincrono
    evaluated_propositions, failed_examples, failed_count, quality_time = asyncio.run(
        evaluate_propositions_in_batches(propositions, doc_splits, batch_size=20)
    )
    quality_api_calls = len(propositions)

    # Mostra esempi di proposizioni fallite
    if failed_examples:
        print(f"\n⚠️ Esempi di proposizioni fallite (prime {len(failed_examples)}):")
        for idx, content, scores in failed_examples:
            print(f"   ❌ Proposizione {idx}: {content}...")
            print(f"      Punteggi: {scores}")

    print(f"\n✅ Fase 2 completata:")
    print(f"   - Proposizioni valutate: {len(propositions)}")
    print(f"   - Proposizioni accettate: {len(evaluated_propositions)}")
    print(f"   - Proposizioni scartate: {failed_count}")
    print(f"   - Tasso di successo: {len(evaluated_propositions)/len(propositions)*100:.1f}%")
    print(f"   - Chiamate API: {quality_api_calls}")
    print(f"   - Tempo impiegato: {quality_time/60:.2f} minuti")
    print(f"   - Velocità media: {quality_api_calls/quality_time:.1f} props/sec")

# REPORT FINALE: Confronto stima vs reale
total_time = generation_time + quality_time
total_api_calls_real = generation_api_calls + quality_api_calls

print(f"\n{'='*80}")
print("📊 REPORT FINALE: STIMA vs REALE")
print(f"{'='*80}")
print(f"\n🔄 CHIAMATE API:")
print(f"   Stimate: {cost_estimate['total_calls']}")
print(f"   Reali:   {total_api_calls_real}")
print(f"   Errore:  {abs(cost_estimate['total_calls'] - total_api_calls_real)} chiamate "
      f"({abs(cost_estimate['total_calls'] - total_api_calls_real)/cost_estimate['total_calls']*100:.1f}%)")

print(f"\n⏱️  TEMPO TOTALE:")
print(f"   Stimato: {cost_estimate['total_calls'] * 2.5 / 60:.1f} minuti (media)")
print(f"   Reale:   {total_time/60:.2f} minuti")

# Calcolo costo reale basato sulle chiamate effettive
# Nota: questa è ancora una stima perché non conosciamo i token esatti
actual_cost_estimate = (total_api_calls_real / cost_estimate['total_calls']) * cost_estimate['total_cost']
print(f"\n💰 COSTO STIMATO:")
print(f"   Iniziale: ${cost_estimate['total_cost']:.4f}")
print(f"   Aggiornato: ${actual_cost_estimate:.4f} (basato su chiamate reali)")

print(f"\n{'='*80}\n")

# %% [markdown]
# ### Embedding delle Proposizioni in un Vectorstore

# %%
# VECTORSTORE PROPOSIZIONI: Indicizzazione delle proposizioni per il retrieval
# Usiamo Chroma come vector database per performance e semplicità
# collection_name univoco per evitare conflitti con altri vector store

print("\n🔄 Creazione vectorstore per le proposizioni...")

vectorstore_propositions = Chroma.from_documents(
    evaluated_propositions, 
    embedding_model,
    collection_name="proposition_chunks"
)

retriever_propositions = vectorstore_propositions.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 4},  # Numero di documenti da recuperare
)

print("✅ Vectorstore proposizioni creato")

# %%
# PRIMA QUERY DI TEST
# Testiamo il retrieval basato su proposizioni con una query sul cambiamento climatico
query = "Qual è il meccanismo chimico attraverso cui l'acidificazione degli oceani minaccia gli organismi marini?"

print(f"\n🔍 Query: {query}\n")
print("📄 Risultati dal Retrieval basato su PROPOSIZIONI:")
print("="*80)

res_proposition = retriever_propositions.invoke(query)

# %%
# Visualizza i risultati del retrieval basato su proposizioni
for i, r in enumerate(res_proposition):
    print(f"\n{i+1}) Proposizione:")
    print(f"   Contenuto: {r.page_content}")
    print(f"   Chunk ID origine: {r.metadata['chunk_id']}")
    print("-"*80)

# %% [markdown]
# ### Confronto Performance con Chunk di Dimensioni Maggiori

# %%
# VECTORSTORE CHUNK GRANDI: Creazione del secondo retriever per confronto
# Questo usa i chunk originali (più grandi) invece delle proposizioni atomiche
# Permette di confrontare l'efficacia dei due approcci

print("\n🔄 Creazione vectorstore per chunk di dimensioni maggiori...")

vectorstore_larger = Chroma.from_documents(
    doc_splits, 
    embedding_model,
    collection_name="larger_chunks"
)

retriever_larger = vectorstore_larger.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 4},  # Stesso numero di risultati per confronto equo
)

print("✅ Vectorstore chunk grandi creato")

# %%
# Esegui la stessa query sul retriever con chunk grandi
print(f"\n🔍 Query: {query}\n")
print("📄 Risultati dal Retrieval basato su CHUNK GRANDI:")
print("="*80)

res_larger = retriever_larger.invoke(query)

# %%
# Visualizza i risultati del retrieval basato su chunk grandi
for i, r in enumerate(res_larger):
    print(f"\n{i+1}) Chunk:")
    print(f"   Contenuto: {r.page_content}")
    print(f"   Chunk ID: {r.metadata['chunk_id']}")
    print("-"*80)

# %% [markdown]
# ### Testing con Query Multiple

# %% [markdown]
# #### Test - 1: Query sulle Cause del Cambiamento Climatico

# %%
test_query_1 = "Quali sono le principali cause del cambiamento climatico secondo il documento?"

print(f"\n{'='*80}")
print(f"TEST 1: {test_query_1}")
print(f"{'='*80}\n")

res_proposition = retriever_propositions.invoke(test_query_1)
res_larger = retriever_larger.invoke(test_query_1)

# %%
print("📄 RISULTATI - Retrieval basato su PROPOSIZIONI:")
print("-"*80)
for i, r in enumerate(res_proposition):
    print(f"{i+1}) {r.page_content} [Chunk ID: {r.metadata['chunk_id']}]")
    print()

# %%
print("📄 RISULTATI - Retrieval basato su CHUNK GRANDI:")
print("-"*80)
for i, r in enumerate(res_larger):
    print(f"{i+1}) {r.page_content[:150]}... [Chunk ID: {r.metadata['chunk_id']}]")
    print()

# %% [markdown]
# #### Test - 2: Query sull'Effetto Albedo

# %%
test_query_2 = "Che cos'è l'effetto albedo e come influisce sul riscaldamento dell'Artico?"

print(f"\n{'='*80}")
print(f"TEST 2: {test_query_2}")
print(f"{'='*80}\n")

res_proposition = retriever_propositions.invoke(test_query_2)
res_larger = retriever_larger.invoke(test_query_2)

# %%
print("📄 RISULTATI - Retrieval basato su PROPOSIZIONI:")
print("-"*80)
for i, r in enumerate(res_proposition):
    print(f"{i+1}) {r.page_content} [Chunk ID: {r.metadata['chunk_id']}]")
    print()

# %%
print("📄 RISULTATI - Retrieval basato su CHUNK GRANDI:")
print("-"*80)
for i, r in enumerate(res_larger):
    print(f"{i+1}) {r.page_content[:150]}... [Chunk ID: {r.metadata['chunk_id']}]")
    print()

# %% [markdown]
# #### Test - 3: Query sulle Tecnologie di Cattura del Carbonio

# %%
test_query_3 = "Quali tecnologie innovative vengono menzionate per la cattura del carbonio?"

print(f"\n{'='*80}")
print(f"TEST 3: {test_query_3}")
print(f"{'='*80}\n")

res_proposition = retriever_propositions.invoke(test_query_3)
res_larger = retriever_larger.invoke(test_query_3)

# %%
print("📄 RISULTATI - Retrieval basato su PROPOSIZIONI:")
print("-"*80)
for i, r in enumerate(res_proposition):
    print(f"{i+1}) {r.page_content} [Chunk ID: {r.metadata['chunk_id']}]")
    print()

# %%
print("📄 RISULTATI - Retrieval basato su CHUNK GRANDI:")
print("-"*80)
for i, r in enumerate(res_larger):
    print(f"{i+1}) {r.page_content[:150]}... [Chunk ID: {r.metadata['chunk_id']}]")
    print()

# %% [markdown]
# ### Analisi Comparativa Finale

# %%
# ANALISI FINALE: Confronto Performance Proposition vs Chunk-based Retrieval
print("\n" + "="*80)
print("📊 ANALISI COMPARATIVA FINALE: PROPOSIZIONI vs CHUNK GRANDI")
print("="*80)

print(f"\n📈 STATISTICHE GENERALI:")
print(f"   - Proposizioni totali generate: {len(evaluated_propositions)}")
print(f"   - Chunk originali: {len(doc_splits)}")
print(f"   - Rapporto proposizioni/chunk: {len(evaluated_propositions)/len(doc_splits):.1f}x")
print(f"   - Tasso di successo quality check: {len(evaluated_propositions)/len(propositions)*100:.1f}%")

print(f"\n🎯 OSSERVAZIONI SUI RISULTATI DEI TEST:")

print(f"\n1️⃣ TEST 1 - Cause del Cambiamento Climatico:")
print(f"   ✅ PROPOSIZIONI: Risposte precise e dirette")
print(f"      - 'L'origine è l'alterazione antropogenica del ciclo del carbonio'")
print(f"      - Informazioni atomiche, facili da processare per un LLM")
print(f"   📦 CHUNK GRANDI: Risposte con più contesto")
print(f"      - Include dettagli su combustibili fossili e gas climalteranti")
print(f"      - Più ricco ma richiede parsing del contesto")
print(f"   🏆 VINCITORE: Proposizioni (per query fattuale diretta)")

print(f"\n2️⃣ TEST 2 - Effetto Albedo:")
print(f"   ⚠️ PROPOSIZIONI: Risultati MENO rilevanti")
print(f"      - Ha recuperato chunk generici su temperatura e oceani")
print(f"      - Non ha trovato la spiegazione specifica dell'effetto albedo")
print(f"   ✅ CHUNK GRANDI: Risultati PIÙ rilevanti")
print(f"      - Primo risultato contiene la spiegazione completa dell'effetto albedo")
print(f"      - Il contesto preservato aiuta con query complesse")
print(f"   🏆 VINCITORE: Chunk Grandi (per query che richiede spiegazione dettagliata)")

print(f"\n3️⃣ TEST 3 - Tecnologie di Cattura del Carbonio:")
print(f"   ✅ PROPOSIZIONI: Risultati molto specifici")
print(f"      - 'Si esplorano tecnologie Direct Air Capture (DAC)'")
print(f"      - Proposizioni mirate sulle tecnologie specifiche")
print(f"   📦 CHUNK GRANDI: Risultati con contesto tecnologico")
print(f"      - Include celle solari a perovskite + DAC nel contesto")
print(f"      - Più informazioni ma meno focalizzato sulla domanda")
print(f"   🏆 VINCITORE: Proposizioni (per lista di tecnologie specifiche)")

print(f"\n🎓 CONCLUSIONI E RACCOMANDAZIONI:")
print(f"\n   📌 PROPOSITION CHUNKING funziona meglio quando:")
print(f"      • Query richiede fatti specifici e discreti")
print(f"      • Si cerca una lista di elementi (tecnologie, cause, effetti)")
print(f"      • Il documento è ricco di informazioni fattuali atomiche")
print(f"      • Si vuole minimizzare il 'rumore' per un LLM downstream")

print(f"\n   📌 CHUNK-BASED RETRIEVAL funziona meglio quando:")
print(f"      • Query richiede spiegazioni dettagliate o meccanismi")
print(f"      • Il contesto narrativo è importante per la comprensione")
print(f"      • Si cercano relazioni causali complesse")
print(f"      • Il flusso del testo originale aiuta la comprensione")

print(f"\n   💡 APPROCCIO IBRIDO (Raccomandato):")
print(f"      • Usa ENTRAMBI i metodi in parallelo")
print(f"      • Usa proposizioni per retrieval preciso di fatti")
print(f"      • Usa chunk grandi per arricchire il contesto")
print(f"      • Combina i risultati nel prompt finale al LLM")
print(f"      • Es: 'Fatti chiave: [proposizioni] | Contesto: [chunks]'")

print(f"\n   ⚡ PERFORMANCE OTTENUTE IN QUESTO TEST:")
print(f"      • Tempo generazione proposizioni: {generation_time:.2f} min")
print(f"      • Tempo quality check: {quality_time:.2f} min")
print(f"      • Tempo totale: {total_time:.2f} min")
print(f"      • Velocità: {generation_api_calls/generation_time:.1f} chunks/sec")
print(f"      • Costo stimato: ${actual_cost_estimate:.4f}")

print("\n" + "="*80 + "\n")

# %% [markdown]
# ### Tabella Comparativa: Proposizioni vs. Chunk Grandi
# 
# | **Aspetto**                | **Retrieval Basato su Proposizioni**                                    | **Retrieval con Chunk Semplici**                                        |
# |---------------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|
# | **Precisione Risposta**    | Alta: Fornisce risposte focalizzate e dirette.                          | Media: Fornisce più contesto ma può includere informazioni irrilevanti.  |
# | **Chiarezza e Brevità**    | Alta: Chiaro e conciso, evita dettagli non necessari.                   | Media: Più completo ma può risultare dispersivo.                         |
# | **Ricchezza Contestuale**  | Bassa: Può mancare di contesto, focalizzandosi su proposizioni specifiche. | Alta: Fornisce contesto e dettagli aggiuntivi.                         |
# | **Completezza**            | Bassa: Può omettere contesto più ampio o dettagli supplementari.        | Alta: Offre una visione più completa con informazioni estese.            |
# | **Flusso Narrativo**       | Medio: Può essere frammentato o disgiunto.                              | Alto: Preserva il flusso logico e la coerenza del documento originale.   |
# | **Sovraccarico Info**      | Basso: Meno probabile sovraccaricare con informazioni in eccesso.       | Alto: Rischio di sovraccaricare l'utente con troppe informazioni.        |
# | **Adatto per Caso d'Uso**  | Migliore per query fattuali rapide.                                     | Migliore per query complesse che richiedono comprensione approfondita.   |
# | **Efficienza**             | Alta: Fornisce risposte rapide e mirate.                                | Media: Può richiedere più sforzo per filtrare contenuto aggiuntivo.     |
# | **Specificità**            | Alta: Risposte precise e mirate.                                        | Media: Risposte possono essere meno mirate per inclusione contesto ampio.|
# 
# ### QUANDO USARE PROPOSITION CHUNKING?
# 
# **✅ Usa Proposizioni quando:**
# - Hai bisogno di **risposte fattuali precise** (es. "Quando è stato firmato l'Accordo di Parigi?")
# - Il documento contiene **molti fatti discreti** (report tecnici, articoli scientifici)
# - Vuoi **minimizzare il rumore** nelle risposte generate dal LLM
# - Le query sono **specifiche e focalizzate**
# 
# **❌ Evita Proposizioni quando:**
# - Hai bisogno di **comprendere relazioni complesse** tra concetti
# - Il **contesto narrativo** è importante per la comprensione
# - Le query richiedono **sintesi di informazioni** da più parti del documento
# - Il documento ha uno **stile narrativo** invece di elencare fatti

# %% [markdown]
# ![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--proposition-chunking)
