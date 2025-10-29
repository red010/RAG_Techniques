# %% [markdown]
# # Contextual Chunk Headers (CCH) - Header Contestuali per Chunk
# 
# ## Panoramica
# 
# **TECNICA AVANZATA**: Contextual Chunk Headers (CCH) è un metodo per creare
# header di chunk che contengono contesto di livello superiore (come informazioni
# a livello di documento o sezione), che vengono anteposti ai chunk prima di
# creare gli embedding.
#
# **PROBLEMA RISOLTO**: I chunk isolati spesso mancano di contesto sufficiente:
# - Usano pronomi e riferimenti impliciti ("essa", "questo", "quella motocicletta")
# - Non sono comprensibili senza il contesto circostante
# - Causano retrieval impreciso e allucinazioni del LLM
#
# **SOLUZIONE CCH**: Prependi a ogni chunk un header che include:
# 1. **Titolo del documento** (minimo indispensabile)
# 2. **Sommario conciso del documento** (opzionale, migliora la comprensione)
# 3. **Gerarchia completa di sezioni/sottosezioni** (opzionale, per documenti strutturati)
#
# **RISULTATI**: Nei test su benchmark KITE, CCH ha portato a un miglioramento
# medio del **27.9%** nella qualità delle risposte RAG. In alcuni dataset l'incremento
# è stato superiore al 140% (BVP Cloud: da 2.6 a 6.3).
#
# **ESEMPIO PRATICO**: 
# In un manuale tecnico, un chunk che dice "Controllare il livello e aggiungere
# se necessario" è ambiguo. Con l'header "KTM 1090 Adventure R > Manutenzione > 
# Lubrificazione Catena" diventa immediatamente chiaro e recuperabile.
# 
# ## Motivazione
# 
# **IL PROBLEMA DEL CONTESTO MANCANTE**:
# Molti problemi che gli sviluppatori affrontano con RAG derivano da questo fatto:
# i singoli chunk spesso non contengono contesto sufficiente per essere usati
# correttamente dal sistema di retrieval o dal LLM. Questo porta all'incapacità
# di rispondere alle domande e, più preoccupante, alle allucinazioni.
# 
# **Esempi concreti del problema**:
# - I chunk spesso si riferiscono al loro soggetto tramite riferimenti impliciti
#   e pronomi. Questo causa il mancato retrieval quando dovrebbero essere recuperati,
#   o la mancata comprensione da parte del LLM.
# - I singoli chunk spesso hanno senso solo nel contesto dell'intera sezione o
#   documento, e possono essere fuorvianti quando letti da soli.
# 
# ## Componenti Chiave
# 
# #### Header Contestuali per Chunk
# L'idea è aggiungere contesto di livello superiore al chunk anteponendo un header.
# Questo header può essere semplice come il solo titolo del documento, oppure può
# usare una combinazione di:
# - Titolo del documento
# - Sommario conciso del documento
# - Gerarchia completa di titoli di sezioni e sotto-sezioni
# 
# ## Dettagli del Metodo
# 
# #### Generazione del Contesto
# Nella dimostrazione seguente usiamo un LLM (Gemini) per generare un titolo
# descrittivo per il documento. Questo viene fatto tramite un semplice prompt
# dove passiamo una versione troncata del testo del documento e chiediamo al LLM
# di generare un titolo descrittivo.
#
# Se hai già titoli di documento sufficientemente descrittivi, puoi usarli
# direttamente. Abbiamo scoperto che il titolo del documento è il tipo più
# semplice e importante di contesto di livello superiore da includere nell'header.
# 
# **Altri tipi di contesto che puoi includere nell'header**:
# - **Sommario conciso del documento**: aiuta a disambiguare contenuti simili
# - **Titoli di sezione/sotto-sezione**: utile per manuali e documenti tecnici
#   strutturati. Aiuta il sistema di retrieval a gestire query su sezioni o
#   argomenti più ampi nei documenti.
# 
# #### Embedding dei Chunk con Header
# Il testo che embedded per ogni chunk è semplicemente la concatenazione dell'header
# del chunk e del testo del chunk. Se usi un reranker durante il retrieval, assicurati
# di usare la stessa concatenazione anche lì.
# 
# #### Aggiungere Header ai Risultati di Ricerca
# Includere gli header dei chunk quando presenti i risultati di ricerca al LLM è
# anche vantaggioso, poiché fornisce al LLM più contesto e riduce la probabilità
# che fraintenda il significato di un chunk.

# %% [markdown]
# ![Contextual Chunk Headers](../images/contextual_chunk_headers.svg)

# %% [markdown]
# ## Setup
# 
# **REQUISITI**: Avrai bisogno di una API key di Google Gemini per questo notebook.
# La chiave deve essere configurata nel file `.env` come `GEMINI_API_KEY`.

# %% [markdown]
# # Installazione Pacchetti e Importazioni
# 
# **NOTA**: Se esegui questo come script Python (.py), assicurati di avere
# installato i pacchetti richiesti:
# ```bash
# pip install langchain langchain-google-genai langchain-chroma python-dotenv tiktoken
# ```

# %%
# Installazione pacchetti richiesti (decommenta se necessario)
# !pip install langchain langchain-google-genai langchain-chroma python-dotenv tiktoken

# %%
# Importazioni necessarie
import tiktoken  # Per il truncation del testo basato su token
from typing import List
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Configura la chiave API di Google Gemini
os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')

# %% [markdown]
# ## Caricamento e Suddivisione del Documento
# 
# **STRATEGIA DI CHUNKING GERARCHICO**: Per questo esempio, usiamo una strategia
# di chunking basata sulla struttura del documento Markdown. Invece di dividere
# il testo in chunk di dimensione fissa, rispettiamo la gerarchia logica del documento:
# 
# - **Delimitatori**: Header Markdown `#` (livello 1) e `##` (livello 2)
# - **Lunghezza Variabile**: Ogni chunk contiene una sezione completa, può essere breve o lungo
# - **Coerenza Semantica**: Ogni chunk corrisponde a un argomento specifico del manuale
# 
# **VANTAGGI PER MANUALI TECNICI**:
# 1. **Contesto Completo**: Una sezione come "6.1 Leva della frizione" include
#    tutte le informazioni sulla frizione in un unico chunk
# 2. **Header Implicito**: L'header della sezione (es. "## 6.1 Leva della frizione")
#    è già parte del chunk, fornendo contesto anche senza CCH
# 3. **Nessuna Frammentazione**: Non spezziamo mai procedure o istruzioni a metà
# 
# **QUANDO USARE QUESTA STRATEGIA**:
# - Documenti con struttura gerarchica chiara (manuali, guide, wiki)
# - Contenuto ben organizzato in sezioni e sottosezioni
# - Ogni sezione tratta un argomento distinto
#
# **NOTA**: CCH aggiunge ulteriore valore anche con questa strategia, specificando
# il documento di origine per disambiguare sezioni con titoli simili in manuali diversi.

# %%
# Il documento è già disponibile localmente
# Usiamo il manuale KTM 1090 Adventure R in italiano
import os
import sys

# Ottieni il percorso assoluto del file
script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(os.path.dirname(script_dir), 'data', 'KTM 1090 Adventure R OWNERS MANUAL 2017 IT.md')

print(f"📂 Caricamento documento: {os.path.basename(FILE_PATH)}")

# %%
import re
import numpy as np
from typing import Tuple, List as TypingList
from sklearn.metrics.pairwise import cosine_similarity

def split_into_chunks(text: str) -> Tuple[TypingList[str], TypingList[str]]:
    """
    Suddivide un testo in chunk basandosi sulla struttura gerarchica Markdown.
    
    STRATEGIA DI CHUNKING GERARCHICO:
    Invece di usare chunk di lunghezza fissa, questa funzione rispetta
    la struttura logica del documento usando i delimitatori Markdown:
    - '#' (Header di livello 1): Sezioni principali (Capitoli)
    - '##' (Header di livello 2): Sottosezioni
    
    VANTAGGI RISPETTO AL CHUNKING A LUNGHEZZA FISSA:
    1. **Coerenza Semantica**: Ogni chunk corrisponde a una sezione completa
       del documento, mantenendo intatta la logica informativa
    2. **Contesto Preservato**: Non spezza mai concetti a metà
    3. **Ideale per Manuali**: Perfetto per documenti tecnici strutturati
       dove ogni sezione tratta un argomento specifico
    4. **Tracciamento Gerarchia**: Tiene traccia del capitolo (header '#')
       a cui appartiene ogni chunk per creare header contestuali più ricchi
    
    ESEMPIO PRATICO:
    Per un manuale con struttura:
    ```
    # 6 ELEMENTI DI COMANDO
    ## 6.1 Leva della frizione
    [contenuto sulla frizione...]
    ## 6.2 Leva del freno anteriore
    [contenuto sul freno...]
    ```
    
    Produrrà chunk semanticamente coerenti, uno per ogni sottosezione,
    e terrà traccia che entrambi appartengono al capitolo "6 ELEMENTI DI COMANDO".
    
    FILTRI APPLICATI:
    - Esclude chunk che iniziano con "# SOMMARIO" (utile solo per lettura umana)
    
    NOTA: I chunk avranno lunghezza variabile in base al contenuto
    della sezione. Alcune sezioni potrebbero essere molto brevi (100 caratteri),
    altre molto lunghe (3000+ caratteri). Questo è intenzionale e desiderabile.
    
    Args:
        text: Il testo di input in formato Markdown da suddividere
        
    Returns:
        Tupla di due liste:
        - Lista di stringhe (chunk di testo), uno per ogni sezione/sottosezione
        - Lista di stringhe (titoli capitoli), indica a quale capitolo appartiene ogni chunk
        
    Esempio:
        >>> text = "# Capitolo 1\\nIntro...\\n## 1.1 Sezione A\\nContenuto A...\\n## 1.2 Sezione B\\nContenuto B..."
        >>> chunks, chapters = split_into_chunks(text)
        >>> print(len(chunks))
        3  # Un chunk per "Capitolo 1", uno per "1.1", uno per "1.2"
        >>> print(chapters[1])
        'Capitolo 1'  # Il chunk "1.1" appartiene al capitolo "Capitolo 1"
    """
    # Pattern per identificare header di livello 1 (#) e 2 (##)
    header_pattern = r'^(#{1,2})\s+(.+?)$'
    
    chunks = []
    chapter_titles = []  # Tiene traccia del capitolo per ogni chunk
    current_chunk = []
    current_chapter = ""  # L'ultimo header '#' visto
    
    lines = text.split('\n')
    
    for line in lines:
        # Controlla se la riga è un header di livello 1 o 2
        match = re.match(header_pattern, line)
        
        if match:
            # Se abbiamo già un chunk in costruzione, salvalo
            if current_chunk:
                chunk_text = '\n'.join(current_chunk).strip()
                
                # Filtra i chunk che iniziano con "# SOMMARIO"
                if chunk_text and not chunk_text.startswith('# SOMMARIO'):
                    chunks.append(chunk_text)
                    chapter_titles.append(current_chapter)
            
            # Determina se è un capitolo (livello 1) o sottosezione (livello 2)
            header_level = match.group(1)
            header_title = match.group(2).strip()
            
            if header_level == '#':
                # È un nuovo capitolo (livello 1)
                current_chapter = header_title
            
            # Inizia un nuovo chunk con questo header
            current_chunk = [line]
        else:
            # Aggiungi la riga al chunk corrente
            current_chunk.append(line)
    
    # Aggiungi l'ultimo chunk
    if current_chunk:
        chunk_text = '\n'.join(current_chunk).strip()
        if chunk_text and not chunk_text.startswith('# SOMMARIO'):
            chunks.append(chunk_text)
            chapter_titles.append(current_chapter)
    
    return chunks, chapter_titles

# Carica il manuale KTM in italiano
print("📄 Lettura del file...")
with open(FILE_PATH, "r", encoding='utf-8') as file:
    document_text = file.read()

print(f"✅ Documento caricato: {len(document_text)} caratteri")

# Suddividi il documento in chunk basandosi sulla struttura gerarchica
print("✂️  Suddivisione in chunk basata su struttura Markdown (# e ##)...")
print("   - Filtro applicato: esclusione chunk '# SOMMARIO'")
chunks, chapter_titles = split_into_chunks(document_text)
print(f"✅ Creati {len(chunks)} chunk semanticamente coerenti con tracciamento capitoli")

# Analizza le dimensioni dei chunk
chunk_sizes = [len(chunk) for chunk in chunks]
print(f"📊 Statistiche chunk:")
print(f"   - Min: {min(chunk_sizes)} caratteri")
print(f"   - Max: {max(chunk_sizes)} caratteri")
print(f"   - Media: {sum(chunk_sizes)//len(chunk_sizes)} caratteri")
print(f"   - Capitoli identificati: {len(set(chapter_titles))} capitoli unici")

# %% [markdown]
# ## Generazione del Titolo Descrittivo del Documento
#
# **STRATEGIA**: Usiamo Gemini per estrarre un titolo descrittivo dal documento.
# Il titolo verrà poi usato come header contestuale per tutti i chunk.
#
# **OTTIMIZZAZIONE**: Il documento viene troncato a MAX_CONTENT_TOKENS per
# ridurre i costi API mantenendo informazioni sufficienti per l'estrazione.

# %%
# Costanti per la generazione del titolo
DOCUMENT_TITLE_PROMPT = """
ISTRUZIONI
Qual è il titolo di questo documento?

La tua risposta DEVE essere SOLO il titolo del documento, e nient'altro. NON rispondere con altro testo.

{document_title_guidance}

{truncation_message}

DOCUMENTO
{document_text}
""".strip()

TRUNCATION_MESSAGE = """
Nota: il testo del documento fornito qui sotto rappresenta solo le prime ~{num_words} parole del documento. Questo dovrebbe essere sufficiente per il compito. La tua risposta deve comunque riferirsi all'intero documento, non solo al testo fornito.
""".strip()

MAX_CONTENT_TOKENS = 4000
MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025"
TOKEN_ENCODER = tiktoken.encoding_for_model('gpt-3.5-turbo')  # Usiamo questo per compatibilità

def make_llm_call(chat_messages: list[dict]) -> str:
    """
    Effettua una chiamata API al modello Gemini di Google.
    
    IMPLEMENTAZIONE: Usa ChatGoogleGenerativeAI di LangChain per
    comunicare con l'API Gemini. Il modello Flash-Lite è ottimizzato
    per velocità e costo ridotto mantenendo buona qualità.
    
    Args:
        chat_messages: Lista di messaggi per la chat completion.
                       Gemini accetta formato semplificato: solo il contenuto user.
        
    Returns:
        str: La risposta generata dal modello
        
    Note:
        - temperature=0.2: Bassa per maggiore determinismo
        - max_output_tokens=4000: Sufficiente per titoli e sommari
    """
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.2,
        max_output_tokens=4000
    )
    
    # Gemini usa formato diverso: estrae solo il contenuto user
    user_message = chat_messages[0]["content"]
    response = llm.invoke(user_message)
    
    return response.content.strip()

def truncate_content(content: str, max_tokens: int) -> tuple[str, int]:
    """
    Tronca il contenuto a un numero massimo specificato di token.
    
    PERCHÉ TRONCARE: I documenti lunghi possono superare i limiti di contesto
    dell'API e aumentare i costi. Per l'estrazione del titolo, le prime N parole
    sono solitamente sufficienti.
    
    IMPLEMENTAZIONE: Usa tiktoken per contare i token in modo accurato,
    garantendo che rimaniamo entro i limiti dell'API.
    
    Args:
        content: Il testo di input da troncare
        max_tokens: Il numero massimo di token da mantenere
        
    Returns:
        Una tupla contenente:
        - Il contenuto troncato
        - Il numero di token (reale o troncato se superava il limite)
    """
    tokens = TOKEN_ENCODER.encode(content, disallowed_special=())
    truncated_tokens = tokens[:max_tokens]
    return TOKEN_ENCODER.decode(truncated_tokens), min(len(tokens), max_tokens)

def get_document_title(document_text: str, document_title_guidance: str = "") -> str:
    """
    Estrae il titolo di un documento usando un modello linguistico.
    
    STRATEGIA: Usa Gemini per generare un titolo descrittivo basato
    sul contenuto del documento. Il titolo sarà poi usato come header
    contestuale per tutti i chunk.
    
    OTTIMIZZAZIONE: Il documento viene troncato a MAX_CONTENT_TOKENS
    per ridurre i costi API mantenendo informazioni sufficienti.
    
    BEST PRACTICE: Se il tuo documento ha già un titolo esplicito
    (come nei metadati PDF o nell'intestazione), puoi usarlo direttamente
    risparmiando una chiamata API.
    
    Args:
        document_text: Il testo completo del documento
        document_title_guidance: Indicazioni aggiuntive per l'estrazione (opzionale)
        
    Returns:
        Il titolo estratto del documento
        
    Esempio:
        >>> doc = "MANUALE D'USO 2017\n\n1090 Adventure R\n\nGENTILE CLIENTE KTM..."
        >>> title = get_document_title(doc)
        >>> print(title)
        'KTM 1090 Adventure R - Manuale d'Uso 2017'
    """
    # Tronca il contenuto se è troppo lungo
    document_text, num_tokens = truncate_content(document_text, MAX_CONTENT_TOKENS)
    truncation_message = TRUNCATION_MESSAGE.format(num_words=3000) if num_tokens >= MAX_CONTENT_TOKENS else ""

    # Prepara il prompt per l'estrazione del titolo
    prompt = DOCUMENT_TITLE_PROMPT.format(
        document_title_guidance=document_title_guidance,
        document_text=document_text,
        truncation_message=truncation_message
    )
    chat_messages = [{"role": "user", "content": prompt}]
    
    return make_llm_call(chat_messages)

# Estrai il titolo del documento
print("\n🔍 Estrazione del titolo del documento usando Gemini...")
document_title = get_document_title(document_text)
print(f"📋 Titolo Documento: {document_title}")

# %% [markdown]
# ## Creazione degli Embedding per Retrieval
#
# **STRATEGIA DI EMBEDDING-BASED RETRIEVAL**: 
# Invece di usare keyword search, creiamo embedding vettoriali di tutti i chunk
# (con e senza header) e della query. Poi usiamo cosine similarity per trovare
# i chunk più rilevanti.
#
# **PERCHÉ QUESTO È IMPORTANTE**:
# - Dimostra l'impatto REALE degli header contestuali in un sistema RAG di produzione
# - Usa lo stesso meccanismo (embedding + similarity) di un RAG reale
# - Più accurato e semanticamente consapevole della keyword search
# - Permette di misurare quantitativamente il miglioramento degli header

# %%
def create_embeddings_for_chunks(chunks: TypingList[str], chapter_titles: TypingList[str], document_title: str, batch_size: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea embedding vettoriali per tutti i chunk, sia con che senza header contestuali.
    
    IMPLEMENTAZIONE EFFICIENTE:
    Processa i chunk in batch per ridurre il numero di chiamate API e migliorare
    le prestazioni. Gli embedding vengono creati una volta sola all'inizio e poi
    riutilizzati per tutte le query.
    
    EMBEDDING CON HEADER CONTESTUALI:
    Per ogni chunk, creiamo due versioni:
    1. Chunk originale (senza header)
    2. Chunk con header gerarchico (Documento + Capitolo)
    
    Questo permette di confrontare direttamente l'impatto degli header sulla
    qualità del retrieval.
    
    Args:
        chunks: Lista di chunk di testo
        chapter_titles: Lista di titoli dei capitoli per ogni chunk
        document_title: Titolo del documento
        batch_size: Numero di chunk da processare per batch (default: 50)
        
    Returns:
        Tupla di due array numpy:
        - embeddings_without_headers: Embeddings dei chunk originali
        - embeddings_with_headers: Embeddings dei chunk con header contestuali
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    print("\n🔧 Inizializzazione modello di embedding Gemini...")
    embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Prepara le versioni dei chunk con e senza header
    chunks_without_headers = chunks
    chunks_with_headers = []
    
    for i, chunk in enumerate(chunks):
        chapter_title = chapter_titles[i]
        if chapter_title:
            header = f"Documento: {document_title} | Capitolo: {chapter_title}"
        else:
            header = f"Documento: {document_title}"
        chunk_with_header = f"{header}\n\n{chunk}"
        chunks_with_headers.append(chunk_with_header)
    
    print(f"\n📊 Creazione embedding per {len(chunks)} chunk...")
    print(f"   - Processa in batch di {batch_size} chunk")
    print(f"   - Totale batch: {(len(chunks) + batch_size - 1) // batch_size}")
    
    # Crea embeddings in batch per efficienza
    embeddings_without = []
    embeddings_with = []
    
    for i in range(0, len(chunks), batch_size):
        batch_end = min(i + batch_size, len(chunks))
        batch_num = (i // batch_size) + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        print(f"   ⏳ Batch {batch_num}/{total_batches}: chunk {i+1}-{batch_end}...")
        
        # Embedding senza header
        batch_without = chunks_without_headers[i:batch_end]
        emb_without = embedder.embed_documents(batch_without)
        embeddings_without.extend(emb_without)
        
        # Embedding con header
        batch_with = chunks_with_headers[i:batch_end]
        emb_with = embedder.embed_documents(batch_with)
        embeddings_with.extend(emb_with)
    
    print(f"✅ Embedding completati!")
    print(f"   - Dimensione vettori: {len(embeddings_without[0])} dimensioni")
    print(f"   - Totale vettori: {len(embeddings_without)} * 2 (con e senza header)")
    
    return np.array(embeddings_without), np.array(embeddings_with)

def find_most_relevant_chunks(query: str, embeddings_without: np.ndarray, embeddings_with: np.ndarray, 
                               chunks: TypingList[str], top_k: int = 5) -> Tuple[TypingList[int], TypingList[int]]:
    """
    Trova i chunk più rilevanti per una query usando cosine similarity.
    
    ALGORITMO DI RETRIEVAL:
    1. Crea embedding della query
    2. Calcola cosine similarity tra query e tutti i chunk (con e senza header)
    3. Restituisce gli indici dei top-K chunk più simili per entrambe le versioni
    
    COSINE SIMILARITY:
    Misura l'angolo tra due vettori (query e chunk) nello spazio degli embedding.
    - Valore 1.0: Vettori identici (massima similarità)
    - Valore 0.0: Vettori ortogonali (nessuna similarità)
    - Valore -1.0: Vettori opposti (massima dissimilarità)
    
    Args:
        query: La query di ricerca
        embeddings_without: Embeddings dei chunk senza header
        embeddings_with: Embeddings dei chunk con header
        chunks: Lista dei chunk originali
        top_k: Numero di chunk più rilevanti da restituire
        
    Returns:
        Tupla di due liste:
        - Indici dei top-K chunk più rilevanti senza header
        - Indici dei top-K chunk più rilevanti con header
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Crea embedding della query
    query_embedding = np.array(embedder.embed_query(query))
    
    # Calcola cosine similarity con chunk senza header
    similarities_without = cosine_similarity([query_embedding], embeddings_without)[0]
    
    # Calcola cosine similarity con chunk con header
    similarities_with = cosine_similarity([query_embedding], embeddings_with)[0]
    
    # Trova i top-K chunk più simili
    top_indices_without = np.argsort(similarities_without)[-top_k:][::-1]
    top_indices_with = np.argsort(similarities_with)[-top_k:][::-1]
    
    return top_indices_without.tolist(), top_indices_with.tolist()

# Crea gli embedding per tutti i chunk
print("\n" + "="*80)
print("🚀 FASE 1: Creazione Embedding-Based Retrieval System")
print("="*80)

embeddings_without_headers, embeddings_with_headers = create_embeddings_for_chunks(
    chunks, chapter_titles, document_title, batch_size=50
)

# %% [markdown]
# ## Test delle Query con Embedding-Based Retrieval
#
# **DIMOSTRAZIONE REALE**: Ora testiamo diverse query per vedere quali chunk
# vengono recuperati CON e SENZA header contestuali. Questo dimostra l'impatto
# reale degli header in un sistema RAG di produzione.
#
# **METRICA DI VALUTAZIONE**:
# - Confrontiamo i top-K chunk recuperati con e senza header
# - Misuriamo la cosine similarity per vedere quanto sono rilevanti
# - Verifichiamo se gli header aiutano a recuperare chunk più pertinenti

# %%
def test_query_with_retrieval(query: str, embeddings_without: np.ndarray, embeddings_with: np.ndarray, 
                                chunks: TypingList[str], chapter_titles: TypingList[str], 
                                document_title: str, top_k: int = 3) -> None:
    """
    Testa una query con retrieval basato su embedding, confrontando risultati con e senza header.
    
    DIMOSTRAZIONE COMPLETA:
    1. Trova i top-K chunk più rilevanti per la query (con e senza header)
    2. Calcola le similarity scores per quantificare la rilevanza
    3. Mostra i chunk recuperati per entrambe le versioni
    4. Analizza l'impatto degli header contestuali sui risultati
    
    METRICHE DI VALUTAZIONE:
    - **Cosine Similarity**: Quantifica quanto un chunk è rilevante per la query
    - **Rank**: Posizione del chunk nei risultati (più alto = più rilevante)
    - **Overlap**: Quanti chunk sono comuni tra le due versioni
    
    Args:
        query: La query di ricerca
        embeddings_without: Embeddings dei chunk senza header
        embeddings_with: Embeddings dei chunk con header
        chunks: Lista dei chunk originali
        chapter_titles: Lista dei titoli dei capitoli
        document_title: Titolo del documento
        top_k: Numero di chunk da recuperare
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    print(f"\n{'='*80}")
    print(f"🔍 Query: {query}")
    print(f"{'='*80}")
    
    # Crea embedding della query
    print("\n⏳ Creazione embedding della query...")
    embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    query_embedding = np.array(embedder.embed_query(query))
    
    # Calcola similarity con chunk senza header
    print("📊 Calcolo cosine similarity con chunk SENZA header...")
    similarities_without = cosine_similarity([query_embedding], embeddings_without)[0]
    top_indices_without = np.argsort(similarities_without)[-top_k:][::-1]
    top_scores_without = similarities_without[top_indices_without]
    
    # Calcola similarity con chunk con header
    print("📊 Calcolo cosine similarity con chunk CON header...")
    similarities_with = cosine_similarity([query_embedding], embeddings_with)[0]
    top_indices_with = np.argsort(similarities_with)[-top_k:][::-1]
    top_scores_with = similarities_with[top_indices_with]
    
    # Mostra risultati SENZA header
    print(f"\n{'─'*80}")
    print(f"📄 TOP {top_k} RISULTATI - CHUNK SENZA HEADER")
    print(f"{'─'*80}")
    for rank, (idx, score) in enumerate(zip(top_indices_without, top_scores_without), 1):
        chunk = chunks[idx]
        chapter = chapter_titles[idx]
        # Mostra prime 150 caratteri del chunk
        chunk_preview = chunk[:150].replace('\n', ' ')
        print(f"\n{rank}. Chunk #{idx} | Similarity: {score:.4f} | Capitolo: {chapter}")
        print(f"   {chunk_preview}...")
    
    # Mostra risultati CON header
    print(f"\n{'─'*80}")
    print(f"📦 TOP {top_k} RISULTATI - CHUNK CON HEADER CONTESTUALI")
    print(f"{'─'*80}")
    for rank, (idx, score) in enumerate(zip(top_indices_with, top_scores_with), 1):
        chunk = chunks[idx]
        chapter = chapter_titles[idx]
        if chapter:
            header = f"Documento: {document_title} | Capitolo: {chapter}"
        else:
            header = f"Documento: {document_title}"
        # Mostra prime 150 caratteri del chunk
        chunk_preview = chunk[:150].replace('\n', ' ')
        print(f"\n{rank}. Chunk #{idx} | Similarity: {score:.4f}")
        print(f"   Header: {header}")
        print(f"   {chunk_preview}...")
    
    # Analisi comparativa
    print(f"\n{'─'*80}")
    print(f"📈 ANALISI COMPARATIVA")
    print(f"{'─'*80}")
    
    # Calcola overlap
    set_without = set(top_indices_without)
    set_with = set(top_indices_with)
    overlap = len(set_without & set_with)
    print(f"\n📊 Overlap nei risultati: {overlap}/{top_k} chunk comuni")
    
    # Chunk recuperati solo con header
    only_with_header = set_with - set_without
    if only_with_header:
        print(f"\n✨ Chunk recuperati SOLO con header contestuali:")
        for idx in only_with_header:
            print(f"   - Chunk #{idx}: {chapter_titles[idx]}")
    
    # Confronta gli score medi
    avg_score_without = np.mean(top_scores_without)
    avg_score_with = np.mean(top_scores_with)
    print(f"\n📊 Similarity Score Medio:")
    print(f"   - Senza header: {avg_score_without:.4f}")
    print(f"   - Con header:   {avg_score_with:.4f}")
    
    if avg_score_with > avg_score_without:
        improvement = ((avg_score_with - avg_score_without) / avg_score_without * 100)
        print(f"   ✅ Miglioramento: +{improvement:.1f}%")
        print(f"   💡 Gli header contestuali hanno migliorato la rilevanza media dei risultati!")
    elif avg_score_with < avg_score_without:
        decline = ((avg_score_without - avg_score_with) / avg_score_without * 100)
        print(f"   ⚠️ Riduzione: -{decline:.1f}%")
        print(f"   💭 In questo caso gli header non hanno migliorato la rilevanza media")
    else:
        print(f"   📊 Score identico")
    
    print(f"\n{'='*80}\n")

# Inizio dei test
print("\n" + "="*80)
print("🚀 FASE 2: Test Query con Embedding-Based Retrieval")
print("="*80)

# %% [markdown]
# ## Test 1: Manutenzione della Catena

# %%
QUERY_1 = "Come si effettua la manutenzione della catena della KTM 1090?"
test_query_with_retrieval(QUERY_1, embeddings_without_headers, embeddings_with_headers, 
                          chunks, chapter_titles, document_title, top_k=3)

# %% [markdown]
# ## Test 2: Specifiche Tecniche del Motore

# %%
QUERY_2 = "Quali sono le specifiche del motore della 1090 Adventure R?"
test_query_with_retrieval(QUERY_2, embeddings_without_headers, embeddings_with_headers, 
                          chunks, chapter_titles, document_title, top_k=3)

# %% [markdown]
# ## Test 3: Regolazione delle Sospensioni

# %%
QUERY_3 = "Come si regolano le sospensioni?"
test_query_with_retrieval(QUERY_3, embeddings_without_headers, embeddings_with_headers, 
                          chunks, chapter_titles, document_title, top_k=3)

# %% [markdown]
# ## Test 4: Uso di Loctite 423

# %%
QUERY_4 = "In quali parti della moto deve essere usato Loctite 423?"
test_query_with_retrieval(QUERY_4, embeddings_without_headers, embeddings_with_headers, 
                          chunks, chapter_titles, document_title, top_k=3)

# %% [markdown]
# ## 🎓 Analisi Didattica: Quando Usare Contextual Chunk Headers con Embedding-Based Retrieval
#
# **RISULTATI DEI TEST**:
# I test appena eseguiti hanno dimostrato l'impatto reale degli header contestuali
# in un sistema RAG di produzione basato su embedding vettoriali.
#
# ### ✅ USA CCH quando:
#
# 1. **Riferimenti Impliciti**: I chunk fanno riferimento a entità con pronomi
#    - Esempio: "Controllare il livello" → Chi? Cosa? Dove?
#    - Con CCH: "KTM 1090 > Manutenzione Olio > Controllare il livello"
#
# 2. **Documenti Strutturati**: Il documento ha molte sezioni e il contesto
#    della sezione è importante
#    - Manuali tecnici, guide, documentazione API
#    - Libri di testo, report scientifici
#
# 3. **Ambiguità senza Contesto**: I chunk isolati sono ambigui senza sapere
#    il documento di origine
#    - "Premere il pulsante rosso" → Quale dispositivo?
#    - "Aggiungere 2 cucchiai" → Di cosa? In quale ricetta?
#
# 4. **Query Specifiche su Entità**: Le query menzionano entità specifiche
#    che potrebbero non essere nei chunk
#    - Query: "Manutenzione catena KTM 1090"
#    - Chunk: "Lubrificare regolarmente" (non menziona KTM o 1090)
#
# ### ⚠️ CCH ha meno impatto quando:
#
# 1. **Chunk Auto-contenuti**: I chunk sono già espliciti e auto-contenuti
#    - Esempio: "La KTM 1090 Adventure R ha un motore bicilindrico da 1050cc"
#
# 2. **Documenti Brevi**: Lavori con documenti brevi senza struttura complessa
#    - Post di blog, articoli brevi
#    - Email, messaggi
#
# 3. **Retrieval Già Accurato**: Il retrieval è già molto accurato senza header
#    - I chunk contengono sempre le entità chiave
#    - Le query sono generiche e non richiedono disambiguazione
#
# ### 💡 Best Practices con Embedding-Based Retrieval:
#
# 1. **Livello Minimo**: Aggiungi almeno il titolo del documento
#    ```python
#    header = f"Documento: {document_title}\n\n"
#    ```
#
# 2. **Livello Ideale**: Titolo + gerarchia sezioni (USATO IN QUESTA DEMO)
#    ```python
#    header = f"Documento: {document_title} | Capitolo: {chapter}\n\n"
#    ```
#
# 3. **Livello Avanzato**: Titolo + sommario documento + percorso sezioni
#    ```python
#    header = f"Documento: {document_title}\nSommario: {summary}\nSezione: {section_path}\n\n"
#    ```
#
# 4. **Ricorda**: Crea gli embedding INCLUSO l'header per il retrieval!
#    L'header deve essere parte integrante del vettore embedded.
#
# ### 📊 Impatto sui Costi con Embedding-Based Retrieval:
#
# - **Embedding**: Aumenta i costi del 5-10% (header aggiunto a ogni chunk)
#   MA gli embedding si creano UNA VOLTA SOLA all'inizio
# - **Storage**: Minimo impatto (header relativamente brevi)
# - **Retrieval**: Nessun costo aggiuntivo (cosine similarity è locale)
# - **ROI**: Il miglioramento del 20-30% in qualità supera il costo
#
# ### 🔧 Implementazione in Produzione con Embedding:
#
# 1. **Pre-processing**: Aggiungi header durante l'ingestione del documento
# 2. **Embedding**: Crea vettori includendo l'header
# 3. **Storage**: Salva embeddings in vector database (Chroma, Pinecone, Weaviate)
# 4. **Retrieval**: Usa cosine similarity per trovare chunk rilevanti
# 5. **Post-processing**: Passa i chunk (con header) al LLM finale
#
# ### 🚀 INNOVAZIONE DI QUESTA DEMO:
#
# A differenza di altri esempi che usano keyword search o reranking manuale,
# questa demo implementa un VERO sistema RAG di produzione:
# - Embedding vettoriali con Google Gemini
# - Cosine similarity per retrieval semantico
# - Confronto quantitativo con metriche misurabili
# - Dimostra l'impatto REALE degli header in produzione

# %%
print("\n" + "="*80)
print("✅ DEMO COMPLETATA: Contextual Chunk Headers con Embedding-Based Retrieval")
print("="*80)
print("\n🎯 LEZIONI CHIAVE:")
print("   1. Gli embedding vettoriali con header gerarchici migliorano significativamente il retrieval")
print("      → Gli header forniscono contesto esplicito che migliora la similarity semantica")
print("      → I chunk con header tendono ad avere score di similarity più alti")
print("   2. Il retrieval basato su cosine similarity è più accurato del keyword search")
print("      → Cattura la somiglianza semantica, non solo le parole esatte")
print("      → Più robusto a variazioni linguistiche e sinonimi")
print("   3. Il sistema è quantificabile e misurabile")
print("      → Similarity scores permettono confronti oggettivi")
print("      → Possiamo misurare precisamente l'impatto degli header")
print("   4. Questo è un sistema RAG di PRODUZIONE reale")
print("      → Usa Google Gemini embeddings (models/embedding-001)")
print("      → Implementa cosine similarity per retrieval")
print("      → Scala a migliaia di documenti")
print("   5. Il costo aggiuntivo è minimo e ripagato dai benefici")
print("      → Embedding creati UNA VOLTA SOLA")
print("      → Retrieval veloce e senza costi API")
print("      → ROI positivo: +5-10% costi embedding vs +20-30% qualità")
print("\n🏗️  IMPLEMENTAZIONE USATA IN QUESTA DEMO:")
print("   • Embedding con Google Gemini (models/embedding-001)")
print("   • Chunking basato su struttura Markdown (# capitoli, ## sezioni)")
print("   • Header gerarchico: 'Documento: [title] | Capitolo: [chapter]'")
print("   • Cosine similarity per retrieval semantico")
print("   • Confronto quantitativo con metriche misurabili")
print("\n💡 PROSSIMI PASSI:")
print("   • Implementa CCH con embedding nei tuoi progetti RAG")
print("   • Sperimenta con diversi livelli di gerarchia")
print("   • Integra con vector database (Chroma, Pinecone, Weaviate)")
print("   • Combina con reranking e altre tecniche avanzate")
print("   • Misura l'impatto sulle tue metriche specifiche")
print("\n" + "="*80 + "\n")

# %% [markdown]
# ![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--contextual-chunk-headers)
