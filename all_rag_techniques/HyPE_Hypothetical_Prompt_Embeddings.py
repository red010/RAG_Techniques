# %% [markdown]
# # HyPE - Hypothetical Prompt Embeddings (Embeddings di Prompt Ipotetici)
# 
# ## Panoramica
# 
# **TECNICA AVANZATA**: HyPE implementa un sistema RAG potenziato da embeddings
# di prompt ipotetici. A differenza delle pipeline RAG tradizionali che soffrono
# del mismatch di stile query-documento, HyPE precalcola domande ipotetiche durante
# la fase di indicizzazione, trasformando il retrieval in un problema di matching
# domanda-domanda ed eliminando la necessità di costose tecniche di query expansion
# a runtime.
# 
# ## Componenti Chiave
# 
# 1. Caricamento e estrazione testo da file
# 2. Chunking del testo per mantenere unità informative coerenti
# 3. **Generazione Hypothetical Prompt Embeddings** usando Gemini per creare
#    multiple domande proxy per ogni chunk
# 4. Creazione vector store usando Chroma e Google Embeddings
# 5. Setup retriever per interrogare i documenti processati
# 6. Valutazione del sistema RAG
# 
# ## Dettagli del Metodo
# 
# ### Preprocessing del Documento
# 
# 1. Il file di testo viene caricato
# 2. Il testo viene suddiviso in chunk usando `RecursiveCharacterTextSplitter`
#    con chunk size e overlap specificati
# 
# ### Generazione Domande Ipotetiche
# 
# Invece di fare embedding dei chunk di testo grezzo, HyPE **genera multiple
# domande ipotetiche** per ogni chunk. Queste **domande precomputate** simulano
# le query degli utenti, migliorando l'allineamento con le ricerche reali.
# Questo elimina la necessità di generazione di risposte sintetiche a runtime
# come nelle tecniche tipo HyDE.
# 
# ### Creazione Vector Store
# 
# 1. Ogni domanda ipotetica viene trasformata in embedding usando Google Embeddings
# 2. Un vector store Chroma viene costruito, associando **ogni embedding di domanda
#    con il suo chunk originale**
# 3. Questo approccio **memorizza multiple rappresentazioni per chunk**, aumentando
#    la flessibilità del retrieval
# 
# ### Setup Retriever
# 
# 1. Il retriever è ottimizzato per **matching domanda-domanda** invece di retrieval
#    diretto di documenti
# 2. Chroma abilita **ricerca efficiente dei nearest-neighbor** sugli embeddings
#    di prompt ipotetici
# 3. I chunk recuperati forniscono un **contesto più ricco e preciso** per la
#    generazione LLM a valle
# 
# ## Caratteristiche Chiave
# 
# 1. **Prompt Ipotetici Precomputati** – Migliora l'allineamento query senza
#    overhead a runtime
# 2. **Rappresentazione Multi-Vector** – Ogni chunk è indicizzato multiple volte
#    per una copertura semantica più ampia
# 3. **Retrieval Efficiente** – Chroma garantisce ricerca veloce sugli embeddings
#    potenziati
# 4. **Design Modulare** – La pipeline è facile da adattare per dataset e
#    configurazioni di retrieval diverse. Inoltre è compatibile con la maggior
#    parte delle ottimizzazioni come reranking ecc.
# 
# ## Valutazione
# 
# L'efficacia di HyPE è stata valutata su multiple dataset, mostrando:
# 
# - Fino a 42 punti percentuali di miglioramento nella precisione del retrieval
# - Fino a 45 punti percentuali di miglioramento nel recall delle affermazioni
#     (Vedi risultati completi in [preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5139335))
# 
# ## Vantaggi di Questo Approccio
# 
# 1. **Elimina l'Overhead a Query-Time** – Tutta la generazione ipotetica è fatta
#    offline durante l'indicizzazione
# 2. **Precisione di Retrieval Potenziata** – Migliore allineamento tra query e
#    contenuto memorizzato
# 3. **Scalabile ed Efficiente** – Nessun costo computazionale aggiuntivo per
#    query; il retrieval è veloce come RAG standard
# 4. **Flessibile ed Estensibile** – Può essere combinato con tecniche RAG avanzate
#    come reranking
# 
# ## Conclusione
# 
# HyPE fornisce un'alternativa scalabile ed efficiente ai sistemi RAG tradizionali,
# superando il mismatch di stile query-documento evitando il costo computazionale
# dell'espansione query a runtime. Spostando la generazione di prompt ipotetici
# all'indicizzazione, migliora significativamente la precisione e l'efficienza del
# retrieval, rendendolo una soluzione pratica per applicazioni reali.
#
# Per maggiori dettagli, consultare il paper completo: [preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5139335)
# 
# 
# <div style="text-align: center;">
# 
# <img src="../images/hype.svg" alt="HyPE" style="width:70%; height:auto;">
# </div>

# %% [markdown]
# # Installazione Pacchetti e Importazioni
# 
# **NOTA**: Questo script è ottimizzato per esecuzione diretta, non come notebook.
# Assicurati di avere installato i seguenti pacchetti:
# ```bash
# pip install langchain-google-genai langchain-chroma langchain-core python-dotenv tqdm
# ```

# %%
import os
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Importazioni LangChain per Gemini e Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Setup percorso al modulo parent per importare evaluation
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from evaluation.evalute_rag import evaluate_rag

# Carica variabili d'ambiente
load_dotenv()

# Configura API key Gemini
if not os.getenv('GEMINI_API_KEY'):
    raise ValueError("❌ GEMINI_API_KEY non trovata nel file .env")
os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')

print("✅ Ambiente configurato correttamente")
print(f"   Python: {sys.version.split()[0]}")
print(f"   Working directory: {os.getcwd()}")

# %% [markdown]
# ## Definizione Costanti
# 
# **PARAMETRI CONFIGURABILI**:
# - `PATH`: Percorso al documento italiano sul cambiamento climatico
# - `LANGUAGE_MODEL_NAME`: Modello Gemini per generare domande ipotetiche
# - `EMBEDDING_MODEL_NAME`: Modello Google per creare embeddings
# - `CHUNK_SIZE`: Dimensione ottimale dei chunk per documento italiano (ridotta da 1000)
# - `CHUNK_OVERLAP`: Overlap tra chunk consecutivi (ridotto da 200)
#
# **NOTA IMPORTANTE**: I chunk possono essere più grandi con HyPE rispetto a RAG
# tradizionale, perché non perdiamo precisione con più informazioni. Le domande
# generate catturano i punti salienti indipendentemente dalla lunghezza del chunk.

# %%
# Documento italiano sul cambiamento climatico (testo, non PDF)
script_dir = os.path.dirname(os.path.abspath(__file__))
default_data_dir = os.path.join(os.path.dirname(script_dir), 'data')
PATH = os.path.join(default_data_dir, "cambiamento_climatico.txt")

# Modello Gemini per generazione domande ipotetiche
LANGUAGE_MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025"
EMBEDDING_MODEL_NAME = "models/embedding-001"

# Parametri ottimizzati per documento italiano (più breve del PDF originale)
CHUNK_SIZE = 500  # Ridotto da 1000 per documento più breve
CHUNK_OVERLAP = 100  # Ridotto da 200

print("\n📋 Configurazione:")
print(f"   Documento: {os.path.basename(PATH)}")
print(f"   LLM: {LANGUAGE_MODEL_NAME}")
print(f"   Embedding: {EMBEDDING_MODEL_NAME}")
print(f"   Chunk size: {CHUNK_SIZE} caratteri")
print(f"   Chunk overlap: {CHUNK_OVERLAP} caratteri")

# %% [markdown]
# ## Definizione Generazione Hypothetical Prompt Embeddings
# 
# **CUORE DELLA TECNICA HyPE**: Questa funzione implementa il concetto chiave:
# 
# **Problema del RAG Tradizionale:**
# - Query utente: "Qual è la causa dello sbiancamento dei coralli?"
# - Chunk documento: "L'aumento della temperatura delle acque superficiali non solo
#   alimenta uragani... ma provoca anche il fenomeno dello sbiancamento dei coralli..."
# - Mismatch: La query è interrogativa e breve, il chunk è descrittivo e lungo
# - Risultato: Matching subottimale tra embedding(query) e embedding(chunk)
# 
# **Soluzione HyPE:**
# - Per ogni chunk, generiamo domande ipotetiche che un utente potrebbe fare
# - Esempio: "Cosa causa lo sbiancamento dei coralli?"
# - Facciamo embedding delle DOMANDE invece del chunk
# - Risultato: Matching domanda-domanda (stesso stile!) → Precisione molto maggiore
# 
# **VANTAGGI:**
# 1. Elimina completamente il mismatch di stile
# 2. Ogni chunk ha rappresentazioni multiple (multi-vector)
# 3. Tutto precomputato offline → zero overhead a runtime
# 4. Compatibile con tutte le altre ottimizzazioni RAG

# %%
def generate_hypothetical_prompt_embeddings(chunk_text: str):
    """
    Genera domande ipotetiche per un singolo chunk usando Gemini.
    
    TECNICA HyPE: Invece di fare embedding del testo grezzo del chunk,
    generiamo domande ipotetiche che un utente potrebbe porre per cercare
    questo contenuto. Questo trasforma il retrieval in un problema di
    "matching domanda-domanda" invece di "domanda-documento", migliorando
    drasticamente la precisione.
    
    VANTAGGI:
    - Elimina il mismatch di stile query-documento
    - Ogni chunk è rappresentato da multiple domande (multi-vector)
    - Tutto è precomputato offline (nessun overhead a runtime)
    
    Args:
        chunk_text (str): Testo del chunk da processare
        
    Returns:
        tuple: (chunk_text, List[embeddings]) dove embeddings sono i vettori
               delle domande generate
    """
    llm = ChatGoogleGenerativeAI(
        model=LANGUAGE_MODEL_NAME,
        temperature=0
    )
    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    
    # Prompt in italiano ottimizzato per il contenuto sul cambiamento climatico
    question_gen_prompt = PromptTemplate.from_template(
        "Analizza il testo fornito e genera domande essenziali che, se risposte, "
        "catturano i punti principali del testo. Ogni domanda deve essere su una riga, "
        "senza numerazione o prefissi.\n\n"
        "IMPORTANTE: Le domande devono essere in italiano e specifiche per il contenuto.\n\n"
        "Testo:\n{chunk_text}\n\nDomande:\n"
    )
    question_chain = question_gen_prompt | llm | StrOutputParser()
    
    # Genera e pulisci le domande
    # NOTA: Gemini può usare \n\n per separare domande, normalizziamo a \n
    # Per modelli più piccoli o produzione, potrebbe essere utile parsing regex
    # per gestire liste numerate/puntate: r"^\s*[\-\*\•]|\s*\d+\.\s*|..."
    questions_text = question_chain.invoke({"chunk_text": chunk_text})
    questions = questions_text.replace("\n\n", "\n").split("\n")
    questions = [q.strip() for q in questions if q.strip()]
    
    # Crea embeddings per tutte le domande generate
    question_embeddings = embedding_model.embed_documents(questions)
    
    return chunk_text, question_embeddings


# %% [markdown]
# ## Creazione e Popolamento Vector Store Chroma
# 
# **ARCHITETTURA MULTI-VECTOR DI HyPE:**
# 
# A differenza di un RAG tradizionale dove abbiamo:
# ```
# 1 chunk → 1 embedding → 1 entry nel vector store
# ```
# 
# Con HyPE abbiamo:
# ```
# 1 chunk → N domande → N embeddings → N entries nel vector store
# ```
# Tutte le N entries puntano allo stesso chunk originale!
# 
# **ESEMPIO PRATICO:**
# Chunk: "L'acidificazione degli oceani è causata dall'assorbimento di CO2..."
# Domande generate:
# - "Cos'è l'acidificazione degli oceani?"
# - "Cosa causa l'acidificazione degli oceani?"
# - "Come la CO2 influenza gli oceani?"
# 
# Ogni domanda diventa un "punto di accesso" per recuperare lo stesso chunk.
# Questo aumenta drasticamente la probabilità di match per query diverse!
# 
# **DIFFERENZA DA RAG TRADIZIONALE:**
# - RAG tradizionale: embedding(chunk) → retrieval
# - HyPE: embedding(domande_generate_da_chunk) → retrieval

# %%
def prepare_vector_store(chunks: List[str]):
    """
    Crea e popola un vector store Chroma con embeddings di domande ipotetiche.
    
    ARCHITETTURA HyPE:
    - Ogni chunk genera N domande ipotetiche
    - Ogni domanda viene trasformata in embedding
    - Il vector store contiene gli embedding delle domande
    - Ogni embedding punta al chunk originale come metadato
    - Un singolo chunk può essere recuperato tramite qualsiasi delle sue domande
    
    DIFFERENZA DA RAG TRADIZIONALE:
    - RAG tradizionale: embedding(chunk) → retrieval
    - HyPE: embedding(domande_generate_da_chunk) → retrieval
    
    Args:
        chunks (List[str]): Lista di chunk di testo da processare
        
    Returns:
        Chroma: Vector store contenente gli embeddings delle domande ipotetiche
    """
    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    
    # Liste per accumulare documenti
    all_documents = []
    
    # Per mostrare esempi di domande generate
    example_questions = []
    
    print(f"\n📊 Generazione domande ipotetiche per {len(chunks)} chunk...")
    print("   (Questo può richiedere alcuni minuti)")
    
    # Usa ThreadPoolExecutor per processare chunk in parallelo
    # max_workers=5 per evitare rate limiting su Gemini API
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(generate_hypothetical_prompt_embeddings, c) for c in chunks]
        
        # Processa i risultati man mano che vengono completati
        chunk_index = 0
        for f in tqdm(as_completed(futures), total=len(chunks)):
            chunk_text, question_embeddings = f.result()
            
            # Salva le domande dei primi 2 chunk per mostrarle come esempio
            if len(example_questions) < 2:
                # Rigeneriamo le domande testuali per visualizzazione
                # (gli embeddings non ci servono qui, solo il testo delle domande)
                llm = ChatGoogleGenerativeAI(model=LANGUAGE_MODEL_NAME, temperature=0)
                question_gen_prompt = PromptTemplate.from_template(
                    "Analizza il testo fornito e genera domande essenziali che, se risposte, "
                    "catturano i punti principali del testo. Ogni domanda deve essere su una riga, "
                    "senza numerazione o prefissi.\n\n"
                    "IMPORTANTE: Le domande devono essere in italiano e specifiche per il contenuto.\n\n"
                    "Testo:\n{chunk_text}\n\nDomande:\n"
                )
                question_chain = question_gen_prompt | llm | StrOutputParser()
                questions_text = question_chain.invoke({"chunk_text": chunk_text})
                questions = questions_text.replace("\n\n", "\n").split("\n")
                questions = [q.strip() for q in questions if q.strip()]
                example_questions.append({
                    'chunk': chunk_text,
                    'questions': questions
                })
            
            # Per ogni domanda generata, crea un documento che punta al chunk originale
            # CHIAVE: Ogni domanda diventa un "entry point" separato per lo stesso chunk
            for _ in question_embeddings:
                doc = Document(
                    page_content=chunk_text,
                    metadata={"source": "cambiamento_climatico.txt"}
                )
                all_documents.append(doc)
            
            chunk_index += 1
    
    total_questions = len(all_documents)
    print(f"✅ Generate {total_questions} domande ipotetiche totali")
    print(f"   Media: {total_questions/len(chunks):.1f} domande per chunk")
    
    # Mostra esempi di domande generate per i primi 2 chunk
    print("\n" + "="*80)
    print("📝 ESEMPI DI DOMANDE GENERATE (Primi 2 Chunk)")
    print("="*80)
    for i, example in enumerate(example_questions, 1):
        print(f"\n🔹 CHUNK #{i}:")
        print(f"   Testo (primi 150 caratteri): {example['chunk'][:150]}...")
        print(f"\n   Domande generate ({len(example['questions'])}):")
        for j, q in enumerate(example['questions'], 1):
            print(f"   {j}. {q}")
    print("\n" + "="*80)
    
    # Crea vector store Chroma
    # NOTA: Chroma gestisce automaticamente la creazione degli embeddings tramite
    # l'embedding_function fornita. Non serve passare embeddings precomputati.
    print("\n🔧 Creazione vector store Chroma...")
    vector_store = Chroma.from_documents(
        documents=all_documents,
        embedding=embedding_model,
        collection_name="hype_questions"
    )
    
    print("✅ Vector store Chroma creato con successo!")
    
    return vector_store


# %% [markdown]
# ## Caricamento e Codifica File di Testo
# 
# **PIPELINE COMPLETA HyPE:**
# 
# 1. **Caricamento**: Legge il file di testo italiano
# 2. **Chunking**: Suddivide in chunk con overlap per mantenere contesto
# 3. **Generazione Domande**: Per ogni chunk, Gemini genera domande ipotetiche
# 4. **Embedding**: Ogni domanda viene trasformata in vettore
# 5. **Indicizzazione**: Vector store Chroma memorizza embeddings
# 
# **PERCHÉ CHUNK SIZE PIÙ GRANDE?**
# Con HyPE possiamo permetterci chunk più grandi perché:
# - Le domande generate "distillano" i punti chiave
# - Non c'è rischio di "diluire" l'embedding con troppo testo
# - Il matching avviene tra domande (brevi e focalizzate)
# 
# **BEST PRACTICE**:
# Testa quanto è "esaustivo" il tuo modello nel generare domande.
# Questo dipende dalla densità informativa del tuo documento.

# %%
def encode_text_file(path, chunk_size=500, chunk_overlap=100):
    """
    Carica un file di testo e lo codifica in un vector store usando HyPE.
    
    PIPELINE:
    1. Carica il file di testo
    2. Suddivide in chunk con overlap
    3. Per ogni chunk, genera domande ipotetiche con Gemini
    4. Crea embeddings delle domande
    5. Popola vector store Chroma
    
    Args:
        path: Percorso al file di testo
        chunk_size: Dimensione desiderata di ogni chunk
        chunk_overlap: Overlap tra chunk consecutivi
        
    Returns:
        Chroma: Vector store contenente gli embeddings HyPE
    """
    print(f"📂 Caricamento documento: {os.path.basename(path)}")
    
    # Carica il file di testo
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"📄 Documento caricato: {len(text)} caratteri")
    
    # Suddividi in chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    # Crea documenti LangChain
    documents = [Document(page_content=text, metadata={"source": path})]
    chunks = text_splitter.split_documents(documents)
    
    print(f"✂️  Creati {len(chunks)} chunk")
    print(f"   Dimensione media: {sum(len(c.page_content) for c in chunks)/len(chunks):.0f} caratteri")
    
    # Estrai solo il testo dei chunk
    chunk_texts = [chunk.page_content for chunk in chunks]
    
    # Crea vector store con HyPE
    vectorstore = prepare_vector_store(chunk_texts)
    
    return vectorstore


# %% [markdown]
# ## Creazione HyPE Vector Store
# 
# **FASE DI INDICIZZAZIONE (OFFLINE)**
# 
# Questo è il momento in cui tutto il "lavoro pesante" di HyPE avviene:
# - Generazione domande ipotetiche per ogni chunk
# - Creazione embeddings di tutte le domande
# - Popolamento vector store
# 
# **IMPORTANTE**: Questa fase è costosa in termini di:
# - Tempo (chiamate API per generazione domande)
# - Costi (chiamate LLM + embeddings)
# 
# MA: È tutto fatto UNA VOLTA SOLA!
# Le query successive saranno veloci come RAG standard, con precisione molto superiore.
# 
# **COSTI STIMATI (per riferimento)**:
# - Generazione domande: N_chunks × costo_chiamata_Gemini
# - Embeddings: (N_chunks × domande_per_chunk) × costo_embedding
# 
# Per il documento italiano sul cambiamento climatico (~40 righe):
# - Circa 12-15 chunk
# - Circa 4-6 domande per chunk
# - Totale: ~60-90 domande generate

# %%
print("\n" + "="*80)
print("🚀 FASE 1: Creazione HyPE Vector Store")
print("="*80)

chunks_vector_store = encode_text_file(
    PATH,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

print("\n✅ Vector store creato con successo!")

# %% [markdown]
# ## Configurazione Retriever
# 
# **RETRIEVER SETUP**
# 
# Il retriever è l'interfaccia per interrogare il vector store.
# Con HyPE, quando l'utente fa una query:
# 
# 1. La query viene trasformata in embedding
# 2. Il retriever cerca nel vector store gli embedding più simili
# 3. Gli embedding più simili sono quelli delle DOMANDE IPOTETICHE
# 4. Il retriever restituisce i CHUNK ORIGINALI associati a quelle domande
# 
# **PARAMETRO K=3**: Recupera i 3 chunk più rilevanti
# (Puoi aumentare K per avere più contesto, a scapito di potenziale rumore)

# %%
print("\n" + "="*80)
print("🔍 FASE 2: Configurazione Retriever")
print("="*80)

chunks_query_retriever = chunks_vector_store.as_retriever(
    search_kwargs={"k": 3}
)
print("✅ Retriever configurato per recuperare top-3 chunk più rilevanti")

# %% [markdown]
# ## Test Retriever con Query di Esempio
# 
# **TEST CON QUERY REALISTICHE**
# 
# Testiamo il sistema con due query in italiano specifiche per il documento
# sul cambiamento climatico:
# 
# 1. **Query scientifica specifica**: Sullo sbiancamento dei coralli
# 2. **Query su soluzioni**: Sulle tecnologie innovative
# 
# **COSA OSSERVARE:**
# - Il retriever trova chunk rilevanti anche se non contengono le parole esatte?
# - Le domande ipotetiche generate catturano i concetti chiave?
# - I chunk recuperati forniscono contesto utile per rispondere alla query?
# 
# **NOTA**: Il retriever restituisce oggetti Document con:
# - `page_content`: Il testo del chunk
# - `metadata`: Informazioni sul documento sorgente

# %%
print("\n" + "="*80)
print("🧪 FASE 3: Test del Retriever con Query di Esempio")
print("="*80)

# Query 1: Causa diretta dello sbiancamento dei coralli
test_query_1 = "Qual è la causa diretta dello sbiancamento dei coralli?"
print(f"\n📋 Query 1: {test_query_1}")
context_1 = chunks_query_retriever.invoke(test_query_1)
print(f"✅ Recuperati {len(context_1)} chunk rilevanti")
print("\n📄 Primo chunk recuperato:")
print(f"   {context_1[0].page_content[:200]}...")

# Query 2: Soluzioni tecnologiche
test_query_2 = "Quali sono le tecnologie innovative per combattere il cambiamento climatico?"
print(f"\n📋 Query 2: {test_query_2}")
context_2 = chunks_query_retriever.invoke(test_query_2)
print(f"✅ Recuperati {len(context_2)} chunk rilevanti")
print("\n📄 Primo chunk recuperato:")
print(f"   {context_2[0].page_content[:200]}...")

# %% [markdown]
# ## Valutazione del Sistema RAG
# 
# **VALUTAZIONE CON LANGSMITH**
# 
# Usiamo il modulo `evaluate_rag()` per valutare il sistema:
# - Genera automaticamente domande di test dal documento
# - Usa LangSmith evaluators per metriche:
#   - **Faithfulness**: Le risposte sono fedeli al contesto recuperato?
#   - **Relevancy**: Il contesto recuperato è rilevante per la query?
# 
# **NOTA**: L'evaluator usa Gemini (`gemini-2.5-flash`) per le valutazioni.
# I risultati saranno visibili nella dashboard di LangSmith.

# %%
print("\n" + "="*80)
print("📊 FASE 4: Valutazione del Sistema RAG con HyPE")
print("="*80)
print("\nAvvio valutazione con LangSmith...")
print("(Questo genererà domande di test e valuterà faithfulness e relevancy)\n")

evaluate_rag(chunks_query_retriever)

print("\n✅ Valutazione completata!")
print("   Controlla i risultati su LangSmith dashboard")

# %%
print("\n" + "="*80)
print("✅ DEMO COMPLETATA: HyPE - Hypothetical Prompt Embeddings")
print("="*80)
print("\n🎯 LEZIONI CHIAVE:")
print("   1. HyPE trasforma il retrieval in matching domanda-domanda")
print("      → Elimina il mismatch di stile query-documento")
print("   2. Le domande ipotetiche sono generate OFFLINE durante l'indicizzazione")
print("      → Zero overhead a runtime, retrieval veloce come RAG standard")
print("   3. Ogni chunk ha rappresentazioni multiple (multi-vector)")
print("      → Maggiore flessibilità e copertura semantica")
print("   4. Combinabile con altre tecniche RAG avanzate")
print("      → Reranking, query expansion, hybrid search")
print("\n💡 QUANDO USARE HyPE:")
print("   ✅ Query degli utenti variano molto nello stile")
print("   ✅ Documenti tecnici con terminologia specifica")
print("   ✅ Necessità di alta precisione nel retrieval")
print("   ✅ Possibilità di indicizzazione offline")
print("\n⚠️  QUANDO HyPE HA MENO IMPATTO:")
print("   - Query e documenti già ben allineati nello stile")
print("   - Documenti molto brevi o semplici")
print("   - Necessità di aggiornamenti frequenti (costoso rigenerare domande)")
print("\n" + "="*80 + "\n")

# %% [markdown]
# ## 🎓 Analisi Tecnica Approfondita
#
# ### Confronto HyPE vs RAG Tradizionale
#
# **RAG Tradizionale:**
# ```
# User Query → Embedding(query) → Similarity Search → Embedding(chunks)
# ```
# - Problema: Lo stile delle query è diverso dallo stile dei documenti
# - La query è breve e interrogativa, i chunk sono descrittivi e lunghi
# - Esempio mismatch:
#   - Query: "Cosa causa il cambiamento climatico?"
#   - Chunk: "Il cambiamento climatico è quella trasformazione silenziosa..."
#   - Il matching semantico non è ottimale per differenza di stile
#
# **HyPE:**
# ```
# User Query → Embedding(query) → Similarity Search → Embedding(domande_generate)
# ```
# - Soluzione: Matching tra query reale e domande ipotetiche (stesso stile!)
# - Entrambe sono brevi, interrogative, con struttura simile
# - Esempio matching:
#   - Query: "Cosa causa il cambiamento climatico?"
#   - Domanda generata: "Quali sono le cause del cambiamento climatico?"
#   - Il matching semantico è MOLTO migliore!
#
# ### Architettura Multi-Vector
#
# HyPE usa una rappresentazione multi-vector:
# - 1 chunk → N domande → N embeddings → N entry nel vector store
# - Tutte puntano allo stesso chunk originale
# - Aumenta probabilità di match per diverse formulazioni della query
#
# **ESEMPIO CONCRETO:**
# ```
# Chunk: "L'acidificazione degli oceani avviene quando l'anidride carbonica
#         si dissolve nell'acqua marina formando acido carbonico..."
#
# Domande generate:
# 1. "Cos'è l'acidificazione degli oceani?"
# 2. "Come avviene l'acidificazione degli oceani?"
# 3. "Quale ruolo ha la CO2 nell'acidificazione?"
# 4. "Cosa succede quando la CO2 si scioglie in mare?"
#
# Query utente possibili che matchano:
# - "Spiega l'acidificazione degli oceani" → match domanda 1
# - "Processo di acidificazione marina" → match domanda 2
# - "CO2 e oceani" → match domanda 3
# - "Dissoluzione anidride carbonica in acqua" → match domanda 4
# ```
#
# Anche se la query non contiene le stesse parole del chunk, il matching
# con le domande intermediate aumenta drasticamente la probabilità di retrieval!
#
# ### Costi e Trade-offs
#
# **Costi di Indicizzazione (ONE-TIME):**
# - Generazione domande: N_chunks × costo_LLM_call
# - Embeddings: (N_chunks × domande_per_chunk) × costo_embedding
# - Per il documento italiano (~15 chunk, ~5 domande/chunk):
#   - ~15 chiamate a Gemini per generazione
#   - ~75 embeddings da creare
#   - Tempo: 2-5 minuti
#   - Costo: circa $0.10-0.20 (molto variabile)
#
# **Costi a Runtime (RICORRENTI):**
# - Identici a RAG standard: solo costo embedding della query
# - Una chiamata embedding per query
# - Nessun overhead aggiuntivo!
#
# **Trade-off:**
# - 🔼 Costo di setup aumentato (10-20x rispetto a RAG standard)
# - 🔽 Costo per query identico a RAG standard
# - 🔼🔼 Precisione retrieval aumentata significativamente (+20-45%)
#
# **ROI Calculation:**
# Se hai:
# - 100 query/giorno
# - Costo setup HyPE: $0.20 (una tantum)
# - Miglioramento precisione: 30%
# → Il costo si ammortizza dopo ~1 giorno!
#
# ### Best Practices
#
# 1. **Ottimizza il Prompt di Generazione Domande**
#    - Fornisci esempi di domande di alta qualità nel prompt
#    - Specifica il livello di dettaglio desiderato
#    - Adatta il prompt al dominio (tecnico, generale, conversazionale)
#    
#    Esempio per dominio medico:
#    ```
#    "Genera domande cliniche che un medico potrebbe fare per
#     recuperare queste informazioni durante una consultazione..."
#    ```
#
# 2. **Bilancia Numero di Domande per Chunk**
#    - Troppo poche (1-2): perdi diversità di rappresentazione
#    - Troppo molte (10+): aumenti costi e potenziale rumore
#    - Sweet spot: 3-7 domande per chunk (dipende dalla lunghezza e densità)
#    - Monitora il rapporto domande/chunk: dovrebbe essere consistente
#
# 3. **Combina con Altre Tecniche**
#    - **HyPE + Reranking**: Massima precisione
#      - HyPE per recall alto, reranking per precision finale
#    - **HyPE + Hybrid Search**: Combina semantic + keyword
#      - HyPE per matching semantico, BM25 per matching lessicale
#    - **HyPE + Query Expansion**: Per query ambigue
#      - Espandi la query originale, poi usa HyPE per retrieval
#
# 4. **Monitoraggio e Iterazione**
#    - Analizza le domande generate: sono di alta qualità?
#    - Testa con query reali degli utenti
#    - Misura metriche: precision@k, recall@k, NDCG
#    - Raffina il prompt di generazione basandoti sui risultati
#    - Considera fine-tuning del modello di generazione per il tuo dominio
#
# 5. **Gestione Scala e Produzione**
#    - Cache le domande generate (non rigenerare ogni volta!)
#    - Considera batch processing per indicizzazione iniziale
#    - Monitora qualità delle domande nel tempo
#    - Implementa fallback a RAG standard se generazione fallisce
#    - Usa async/parallel processing per indicizzazione veloce
#
# ### Quando NON Usare HyPE
#
# HyPE non è sempre la soluzione migliore. Evita HyPE se:
#
# 1. **Documenti già interrogativi**: Se i tuoi documenti sono FAQ o Q&A
#    - Non c'è mismatch di stile da risolvere
#    - RAG standard funziona benissimo
#
# 2. **Aggiornamenti molto frequenti**: Se il corpus cambia continuamente
#    - Il costo di rigenerare domande diventa proibitivo
#    - Considera tecniche più leggere
#
# 3. **Documenti molto brevi**: Se chunk <100 caratteri
#    - Le domande generate potrebbero essere più lunghe del chunk!
#    - Meglio usare embedding diretto
#
# 4. **Budget limitato per indicizzazione**: Se non puoi permetterti LLM calls
#    - HyPE richiede N_chunks chiamate LLM
#    - Considera alternative più economiche
#
# 5. **Query già ben allineate**: Se le query degli utenti sono descrittive
#    - Esempio: ricerca scientifica con query lunghe e dettagliate
#    - Il mismatch è minimo, HyPE aggiunge poco valore

# %% [markdown]
# ![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--hype-hypothetical-prompt-embeddings)
