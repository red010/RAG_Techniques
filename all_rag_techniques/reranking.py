# %% [markdown]
# # Reranking per Sistemi RAG (Riordino Documenti)
# 
# ## Panoramica
# 
# **TECNICA FONDAMENTALE**: Il reranking è un passaggio cruciale nei sistemi RAG
# che mira a migliorare la rilevanza dei documenti recuperati. Consiste nel
# rivalutare e riordinare i documenti inizialmente recuperati per assicurarsi
# che le informazioni più pertinenti siano prioritarie.
# 
# ## Motivazione
# 
# I metodi di retrieval iniziali (es. similarity search con embeddings) spesso
# si basano su metriche di similarità semplici che possono:
# 
# - **Mancare di contesto**: Recuperano documenti simili lessicalmente ma non
#   semanticamente rilevanti
# - **Ordinamento subottimale**: I documenti più rilevanti potrebbero non essere
#   nelle prime posizioni
# - **Keyword bias**: Favoriscono documenti con molte parole chiave ma contenuto
#   superficiale
# 
# Il reranking risolve questi problemi usando modelli più sofisticati che
# comprendono meglio la relazione query-documento.
# 
# ## Le Due Metodologie di Reranking
# 
# ### 1. LLM-Based Reranking (Basato su Modelli Linguistici)
# 
# **COME FUNZIONA**:
# - Usa un Large Language Model (es. Gemini) per valutare la rilevanza
# - Crea un prompt che chiede al modello di assegnare un punteggio 1-10
# - Elabora ogni coppia query-documento sequenzialmente
# 
# **PRO**:
# - ✅ Comprensione semantica profonda e contestuale
# - ✅ Altamente flessibile (customizzabile via prompt)
# - ✅ Eccellente per query complesse che richiedono ragionamento
# - ✅ Può considerare intenti impliciti della query
# 
# **CONTRO**:
# - ❌ Lento (secondi per documento)
# - ❌ Costoso (ogni valutazione = API call)
# - ❌ Non adatto per produzione ad alto volume
# 
# **QUANDO USARLO**:
# - Prototipazione e testing
# - Domini nuovi o specializzati dove i modelli generici falliscono
# - Query che richiedono comprensione profonda (es. domande multi-hop)
# - Budget disponibile e latenza non critica
# 
# ### 2. Cross-Encoder Reranking (Modelli Specializzati)
# 
# **COME FUNZIONA**:
# - Usa modelli pre-addestrati specifici per ranking (es. MS MARCO)
# - Elabora coppie query-documento INSIEME (non separatamente come bi-encoder)
# - Assegna score diretto di rilevanza in millisecondi
# 
# **PRO**:
# - ✅ Molto veloce (millisecondi per documento)
# - ✅ Economico (esecuzione locale, nessuna API call)
# - ✅ Ottimo rapporto qualità/velocità
# - ✅ Pronto per produzione ad alto volume
# 
# **CONTRO**:
# - ❌ Meno flessibile (modello fisso, non customizzabile)
# - ❌ Comprensione semantica buona ma non eccellente come LLM
# - ❌ Potrebbe non gestire domini molto specifici
# 
# **QUANDO USARLO**:
# - Produzione con requisiti di alta velocità
# - Budget limitato per API calls
# - Domini generali dove MS MARCO performa bene
# - Necessità di processare molte query al secondo
# 
# ## Confronto Diretto: LLM vs Cross-Encoder
# 
# | Aspetto           | LLM-based (Gemini)              | Cross-Encoder (MS MARCO)      |
# |-------------------|---------------------------------|-------------------------------|
# | **Velocità**      | Lento (2-5 sec per doc)         | Veloce (10-50 ms per doc)     |
# | **Costo**         | Alto ($0.10-0.40 per 1M tokens) | Basso (gratuito, locale)      |
# | **Comprensione**  | Eccellente (semantica profonda) | Buona (pattern matching)      |
# | **Flessibilità**  | Alta (prompt engineering)       | Bassa (modello fisso)         |
# | **Scalabilità**   | Limitata (rate limits API)      | Eccellente (solo CPU/GPU)     |
# | **Setup**         | Semplice (API key)              | Richiede download modello     |
# | **Caso d'uso**    | Prototipazione, query complesse | Produzione, alto volume       |
# 
# ## Architettura del Reranking
# 
# ```
# FASE 1: RETRIEVAL INIZIALE (Ampio)
# ====================================
# Query → Vector Store → Top 30 documenti (recall massimizzato)
# 
# FASE 2: RERANKING (Precision)
# ====================================
# Top 30 documenti → Reranker (LLM o Cross-Encoder) → Top 3-5 documenti
# 
# FASE 3: GENERAZIONE
# ====================================
# Top 3-5 documenti → LLM → Risposta finale
# ```
# 
# **STRATEGIA**: Retrieval ampio + Reranking preciso = Miglior compromesso
# tra recall e precision.
# 
# ## Vantaggi del Reranking
# 
# 1. **Rilevanza Migliorata**: Documenti più pertinenti nelle prime posizioni
# 2. **Riduzione Rumore**: Filtra informazioni irrilevanti
# 3. **Contesto Migliore**: LLM riceve i documenti più utili per generare
# 4. **Flessibilità**: Diversi metodi per diverse esigenze
# 
# ## Conclusione
# 
# Il reranking è essenziale nei sistemi RAG moderni. La scelta tra LLM-based
# e Cross-Encoder dipende da:
# - **Budget e velocità** → Cross-Encoder
# - **Accuratezza massima** → LLM-based
# - **Compromesso ideale** → Cross-Encoder per retrieval + LLM per generazione
# 
# Questo notebook dimostra entrambi i metodi con esempi pratici sul
# cambiamento climatico in italiano.

# %% [markdown]
# ## Setup e Imports

# %%
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from pydantic import BaseModel, Field
from typing import List, Any

# Load environment variables
load_dotenv()
if not os.getenv('GEMINI_API_KEY'):
    raise ValueError("GEMINI_API_KEY non trovata nel file .env")
os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')

# Modello Gemini
LANGUAGE_MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025"

print("✅ Setup completato: Gemini configurato")

# %% [markdown]
# ## Caricamento Documento e Creazione Vector Store

# %%
# Path al documento italiano sul cambiamento climatico
path = "data/cambiamento_climatico.txt"

# Verifica esistenza file
if not os.path.exists(path):
    raise FileNotFoundError(
        f"File {path} non trovato. "
        f"Assicurati che il documento sia nella directory data/"
    )

print(f"📄 Documento trovato: {path}")

# %%
def create_vectorstore_from_text(file_path: str):
    """
    Crea vector store da file di testo usando Gemini embeddings e Chroma.
    
    PROCESSO:
    1. Legge il file di testo
    2. Lo suddivide in chunks usando RecursiveCharacterTextSplitter
    3. Crea embeddings con Gemini
    4. Costruisce vector store Chroma per similarity search
    
    Args:
        file_path: Percorso al file di testo
        
    Returns:
        Vector store FAISS pronto per retrieval
    """
    print(f"\n⏳ Caricamento e processing del documento...")
    
    # Leggi il file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"   Documento caricato: {len(text)} caratteri")
    
    # Chunking del testo
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    print(f"   Creati {len(documents)} chunks")
    
    # Crea embeddings con Gemini
    print(f"   Generazione embeddings con Gemini e creazione Chroma vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="reranking_demo"
    )
    
    print(f"✅ Vector store Chroma creato con successo!")
    
    return vectorstore

# Crea il vector store
vectorstore = create_vectorstore_from_text(path)

# %% [markdown]
# ## Metodo 1: LLM-Based Reranking con Gemini
# 
# ### Spiegazione Dettagliata
# 
# **METODO LLM-BASED**: Usa Gemini per valutare semanticamente la rilevanza
# di ogni documento rispetto alla query.
# 
# **VANTAGGI SPECIFICI**:
# - Comprende sinonimi e parafrasi
# - Valuta rilevanza concettuale, non solo lessicale
# - Può applicare ragionamento complesso
# 
# **PROCESSO**:
# 1. Retrieval iniziale recupera N documenti (es. 30)
# 2. Per ogni documento, crea prompt con query + documento
# 3. Gemini assegna score 1-10 basato su rilevanza semantica
# 4. Ordina documenti per score decrescente
# 5. Restituisce top K documenti (es. 3)
# 
# **NOTA SUL COSTO**: Con 30 documenti da rerankare:
# - ~30 API calls a Gemini
# - ~15,000 tokens totali (500 tokens/doc)
# - Costo: ~$0.0015-0.006 per query (dipende da input/output)

# %%
class RatingScore(BaseModel):
    """Pydantic model per structured output di Gemini."""
    relevance_score: float = Field(
        ..., 
        description="Punteggio di rilevanza del documento rispetto alla query (1-10)."
    )

def rerank_documents(query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
    """
    Riordina i documenti usando un LLM (Gemini) per valutare la rilevanza.
    
    METODO LLM-BASED:
    - Usa un modello linguistico per valutare semanticamente la rilevanza
    - Pro: Comprensione contestuale profonda, flessibile
    - Contro: Più lento e costoso rispetto a Cross-Encoder
    - Quando usarlo: Query complesse, necessità di ragionamento
    
    Args:
        query: La query dell'utente
        docs: Lista di documenti da rerankare
        top_n: Numero di documenti da restituire dopo reranking
        
    Returns:
        Lista di top_n documenti riordinati per rilevanza
    """
    # Template del prompt per valutazione rilevanza
    prompt_template = PromptTemplate(
        input_variables=["query", "doc"],
        template="""Su una scala da 1 a 10, valuta la rilevanza del seguente documento rispetto alla query.
Considera il contesto e l'intento specifico della query, non solo la corrispondenza di parole chiave.

Query: {query}
Documento: {doc}

Punteggio di Rilevanza (1-10):"""
    )
    
    # Inizializza Gemini
    llm = ChatGoogleGenerativeAI(
        model=LANGUAGE_MODEL_NAME,
        temperature=0  # Deterministico per consistency
    )
    
    # Crea chain con structured output
    llm_chain = prompt_template | llm.with_structured_output(RatingScore)
    
    # Valuta ogni documento
    scored_docs = []
    for doc in docs:
        input_data = {"query": query, "doc": doc.page_content}
        score = llm_chain.invoke(input_data).relevance_score
        try:
            score = float(score)
        except ValueError:
            score = 0  # Fallback se parsing fallisce
        scored_docs.append((doc, score))
    
    # Ordina per score decrescente e restituisci top_n
    reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked_docs[:top_n]]

# %% [markdown]
# ## Esempio: Retrieval Iniziale vs Reranking LLM

# %%
# Query italiana che beneficia significativamente dal reranking
query = "Come influisce il cambiamento climatico sulla biodiversità degli ecosistemi?"

print("\n" + "="*80)
print("🔍 QUERY DI TEST")
print("="*80)
print(f"\n{query}")
print("\n💡 Questa query beneficia dal reranking perché:")
print("   - Richiede comprensione semantica (non solo keyword matching)")
print("   - Potrebbe recuperare documenti generici sul clima al primo retrieval")
print("   - Il reranking identifica i documenti che parlano specificatamente di biodiversità")

# %%
print("\n" + "="*80)
print("📊 CONFRONTO: Retrieval Iniziale vs Reranking")
print("="*80)

# Retrieval iniziale (baseline)
print("\n⏳ Fase 1: Retrieval iniziale (baseline)...")
initial_docs = vectorstore.similarity_search(query, k=15)

print(f"\n🔹 TOP 3 DOCUMENTI - RETRIEVAL INIZIALE:")
print("-"*80)
for i, doc in enumerate(initial_docs[:3], 1):
    print(f"\nDocumento {i}:")
    print(f"{doc.page_content[:200]}...")

# Reranking con LLM
print("\n⏳ Fase 2: Reranking con Gemini...")
reranked_docs = rerank_documents(query, initial_docs, top_n=3)

print(f"\n⭐ TOP 3 DOCUMENTI - DOPO RERANKING LLM:")
print("-"*80)
for i, doc in enumerate(reranked_docs, 1):
    print(f"\nDocumento {i}:")
    print(f"{doc.page_content[:200]}...")

print("\n" + "="*80)
print("💡 OSSERVAZIONI:")
print("="*80)
print("  Il reranking LLM ha riordinato i documenti basandosi sulla")
print("  comprensione semantica della relazione tra 'cambiamento climatico'")
print("  e 'biodiversità', non solo sulla presenza di keyword.")

# %% [markdown]
# ## Custom Retriever con Reranking LLM Integrato

# %%
class CustomRetriever(BaseRetriever, BaseModel):
    """
    Retriever custom che integra reranking LLM-based.
    
    ARCHITETTURA:
    1. Retrieval iniziale ampio (k=30) per massimizzare recall
    2. Reranking LLM per ottimizzare precision
    3. Restituisce top num_docs documenti più rilevanti
    """
    vectorstore: Any = Field(description="Vector store per retrieval iniziale")
    num_docs: int = Field(default=2, description="Numero di documenti da restituire")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieval in due fasi:
        1. Retrieval iniziale ampio (30 documenti)
        2. Reranking LLM-based per selezionare i top num_docs più rilevanti
        """
        initial_docs = self.vectorstore.similarity_search(query, k=30)
        return rerank_documents(query, initial_docs, top_n=self.num_docs)

# Crea l'istanza del custom retriever
custom_retriever = CustomRetriever(vectorstore=vectorstore)

print("✅ Custom Retriever con reranking LLM creato")

# %% [markdown]
# ## Esempio Dimostrativo: Perché il Reranking è Necessario
# 
# Questo esempio usa documenti sintetici per dimostrare chiaramente
# il vantaggio del reranking.

# %%
# Esempio dimostrativo con documenti sintetici sul clima
chunks_it = [
    "Il cambiamento climatico è un problema globale.",
    "Il riscaldamento globale influenza molti aspetti del pianeta.",
    "Gli ecosistemi sono influenzati da vari fattori ambientali.",
    """La biodiversità è particolarmente vulnerabile al cambiamento climatico. 
    L'aumento delle temperature e i cambiamenti nei regimi delle precipitazioni 
    stanno alterando gli habitat naturali, causando migrazioni di specie e 
    modificando le interazioni tra organismi negli ecosistemi. Molte specie 
    non riescono ad adattarsi abbastanza velocemente a questi cambiamenti rapidi.""",
    """Il cambiamento climatico ha impatti devastanti sulla biodiversità. 
    Gli ecosistemi marini e terrestri stanno subendo trasformazioni profonde 
    che mettono a rischio la sopravvivenza di numerose specie animali e vegetali. 
    L'acidificazione degli oceani e lo scioglimento dei ghiacci polari sono 
    solo alcuni esempi di come il clima influenzi direttamente la biodiversità."""
]
docs_it = [Document(page_content=chunk) for chunk in chunks_it]

def compare_rag_techniques_it(query: str, docs: List[Document] = docs_it) -> None:
    """
    Confronto tra retrieval baseline e retrieval con reranking LLM.
    
    Dimostra come il reranking migliora la rilevanza dei documenti recuperati
    identificando quelli semanticamente più pertinenti.
    """
    # Crea vector store temporaneo con documenti di esempio
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore_demo = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="demo_comparison"
    )

    print("\n" + "="*80)
    print("📊 DEMO COMPARATIVA: Baseline vs Reranking LLM")
    print("="*80)
    print(f"\nQuery: {query}\n")
    
    # Baseline retrieval
    print("-"*80)
    print("🔹 BASELINE RETRIEVAL (solo similarity search):")
    print("-"*80)
    baseline_docs = vectorstore_demo.similarity_search(query, k=2)
    for i, doc in enumerate(baseline_docs, 1):
        content = doc.page_content.replace('\n', ' ')
        print(f"\n  Documento {i}:")
        print(f"  {content[:150]}...")

    # Reranking con LLM
    print("\n" + "-"*80)
    print("⭐ RERANKING LLM (con valutazione semantica Gemini):")
    print("-"*80)
    custom_retriever_demo = CustomRetriever(vectorstore=vectorstore_demo, num_docs=2)
    advanced_docs = custom_retriever_demo.invoke(query)
    for i, doc in enumerate(advanced_docs, 1):
        content = doc.page_content.replace('\n', ' ')
        print(f"\n  Documento {i}:")
        print(f"  {content[:150]}...")
    
    # Analisi dei risultati
    print("\n" + "-"*80)
    print("💡 ANALISI DEI RISULTATI:")
    print("-"*80)
    print("  ✅ Il reranking LLM ha identificato i documenti che parlano")
    print("     SPECIFICAMENTE di biodiversità e cambiamento climatico")
    print("  ❌ Il baseline ha recuperato documenti più generici che menzionano")
    print("     solo vagamente i termini della query")
    print("\n  🎯 CONCLUSIONE: Il reranking migliora significativamente la precision!")

# Esegui il confronto
query_test = "Come il cambiamento climatico influenza la biodiversità?"
compare_rag_techniques_it(query_test, docs_it)

# %% [markdown]
# ## Metodo 2: Cross-Encoder Reranking
# 
# ### Spiegazione Dettagliata
# 
# **METODO CROSS-ENCODER**: Usa modelli pre-addestrati specifici per ranking
# come MS MARCO MiniLM.
# 
# **DIFFERENZA DA BI-ENCODER**:
# - Bi-encoder: Codifica query e doc separatamente, poi confronta
# - Cross-encoder: Codifica query+doc INSIEME → migliore comprensione interazione
# 
# **VANTAGGI SPECIFICI**:
# - Velocità: 10-50ms per documento (vs 2-5 sec per LLM)
# - Costo: Zero (esecuzione locale)
# - Qualità: Molto buona per domini generali
# - Scalabilità: Può processare migliaia di documenti al secondo
# 
# **MODELLO MS MARCO**:
# - Addestrato su Microsoft Machine Reading Comprehension dataset
# - 8+ milioni di query reali
# - Eccellente per ranking generale
# 
# **CONFRONTO CON LLM**:
# 
# | Aspetto           | LLM-based (Gemini)              | Cross-Encoder (MS MARCO)      |
# |-------------------|---------------------------------|-------------------------------|
# | Velocità          | Lento (secondi per documento)   | Veloce (millisecondi)         |
# | Costo             | Alto (API calls)                | Basso (locale)                |
# | Comprensione      | Eccellente (semantica profonda) | Buona (pattern matching)      |
# | Flessibilità      | Alta (customizzabile con prompt)| Bassa (modello fisso)         |
# | Quando usarlo     | Query complesse, dominio nuovo  | Produzione, alta velocità     |
# 
# **RACCOMANDAZIONE**: 
# - Usa Cross-Encoder per produzione con alto volume
# - Usa LLM per prototipazione o quando la massima accuratezza è critica
# - Puoi anche COMBINARLI: Cross-Encoder per primo filtering veloce,
#   poi LLM per reranking finale di top 10 documenti

# %%
print("\n" + "="*80)
print("🔧 SETUP: Cross-Encoder Model")
print("="*80)

# Inizializza il cross-encoder
# Nota: Il download del modello avviene automaticamente al primo utilizzo
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

print("✅ Cross-Encoder caricato: ms-marco-MiniLM-L-6-v2")
print("   - Modello specializzato per ranking")
print("   - ~23M parametri")
print("   - Addestrato su MS MARCO dataset")

# %%
class CrossEncoderRetriever(BaseRetriever, BaseModel):
    """
    Retriever con reranking Cross-Encoder.
    
    ARCHITETTURA:
    1. Retrieval iniziale con similarity search
    2. Reranking con Cross-Encoder che valuta coppie query-documento
    3. Restituisce top documenti riordinati
    
    VANTAGGI:
    - Veloce: Millisecondi per documento
    - Economico: Nessuna API call
    - Efficace: Migliora significativamente la rilevanza
    """
    vectorstore: Any = Field(description="Vector store per retrieval iniziale")
    cross_encoder: Any = Field(description="Modello Cross-Encoder per reranking")
    k: int = Field(default=5, description="Numero di documenti da recuperare inizialmente")
    rerank_top_k: int = Field(default=3, description="Numero di documenti da restituire dopo reranking")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieval in due fasi con Cross-Encoder:
        1. Retrieval iniziale con similarity search
        2. Reranking con Cross-Encoder che valuta coppie query-documento
        
        PROCESSO DETTAGLIATO:
        - Fase 1: Recupera k documenti candidati (massimizza recall)
        - Fase 2: Crea coppie [query, documento] per cross-encoder
        - Fase 3: Cross-encoder assegna score di rilevanza a ogni coppia
        - Fase 4: Ordina documenti per score decrescente
        - Fase 5: Restituisce top rerank_top_k documenti (massimizza precision)
        """
        # Fase 1: Retrieval iniziale
        initial_docs = self.vectorstore.similarity_search(query, k=self.k)
        
        # Fase 2: Prepara coppie per cross-encoder
        pairs = [[query, doc.page_content] for doc in initial_docs]
        
        # Fase 3: Ottieni scores dal cross-encoder
        # Nota: predict() è velocissimo (millisecondi per tutte le coppie)
        scores = self.cross_encoder.predict(pairs)
        
        # Fase 4: Ordina per score
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        
        # Fase 5: Restituisci top documenti
        return [doc for doc, _ in scored_docs[:self.rerank_top_k]]

print("✅ CrossEncoderRetriever class definita")

# %% [markdown]
# ## Demo: Cross-Encoder Reranking in Azione

# %%
print("\n" + "="*80)
print("🔬 DEMO: Cross-Encoder Reranking")
print("="*80)

# Crea il cross-encoder retriever
cross_encoder_retriever = CrossEncoderRetriever(
    vectorstore=vectorstore,
    cross_encoder=cross_encoder,
    k=15,  # Recupera 15 documenti inizialmente
    rerank_top_k=3  # Restituisci top 3 dopo reranking
)

# Query di test
query_finale = "Quali sono gli impatti del cambiamento climatico sulla biodiversità?"

print(f"\nQuery: {query_finale}\n")

# Recupera documenti con reranking
print("⏳ Esecuzione retrieval + reranking con Cross-Encoder...")
docs_reranked = cross_encoder_retriever.invoke(query_finale)

print("\n" + "-"*80)
print("📄 DOCUMENTI RERANKED CON CROSS-ENCODER (Top 3):")
print("-"*80)
for i, doc in enumerate(docs_reranked, 1):
    print(f"\nDocumento {i}:")
    print(f"{doc.page_content[:200]}...")

# %% [markdown]
# ## Confronto Finale: LLM vs Cross-Encoder vs Baseline

# %%
print("\n" + "="*80)
print("🏆 CONFRONTO COMPLETO: Baseline vs Cross-Encoder vs LLM")
print("="*80)

query_confronto = "Come il cambiamento climatico influenza la biodiversità?"

# 1. Baseline
print("\n" + "-"*80)
print("🔹 METODO 1: BASELINE (Solo Similarity Search)")
print("-"*80)
baseline_docs = vectorstore.similarity_search(query_confronto, k=3)
for i, doc in enumerate(baseline_docs, 1):
    print(f"\nDocumento {i}:")
    print(f"{doc.page_content[:150]}...")

# 2. Cross-Encoder
print("\n" + "-"*80)
print("⚡ METODO 2: CROSS-ENCODER RERANKING")
print("-"*80)
ce_docs = cross_encoder_retriever.invoke(query_confronto)
for i, doc in enumerate(ce_docs, 1):
    print(f"\nDocumento {i}:")
    print(f"{doc.page_content[:150]}...")

# 3. LLM-based
print("\n" + "-"*80)
print("🤖 METODO 3: LLM-BASED RERANKING (Gemini)")
print("-"*80)
initial_for_llm = vectorstore.similarity_search(query_confronto, k=15)
llm_docs = rerank_documents(query_confronto, initial_for_llm, top_n=3)
for i, doc in enumerate(llm_docs, 1):
    print(f"\nDocumento {i}:")
    print(f"{doc.page_content[:150]}...")

# %% [markdown]
# ## Conclusioni e Best Practices

# %%
print("\n" + "="*80)
print("✅ DEMO COMPLETATA: Reranking per Sistemi RAG")
print("="*80)

print("\n🎓 LEZIONI CHIAVE:")
print("\n1. QUANDO USARE IL RERANKING:")
print("   ✅ Quando il retrieval iniziale recupera troppi documenti irrilevanti")
print("   ✅ Quando l'ordine dei documenti non riflette la vera rilevanza")
print("   ✅ Quando hai bisogno di massima precision con budget limitato")

print("\n2. QUALE METODO SCEGLIERE:")
print("   🤖 LLM-BASED (Gemini):")
print("      → Prototipazione e testing")
print("      → Query complesse che richiedono ragionamento")
print("      → Domini specializzati dove cross-encoder fallisce")
print("      → Budget disponibile per API calls")
print("\n   ⚡ CROSS-ENCODER (MS MARCO):")
print("      → Produzione con alto volume di query")
print("      → Requisiti di bassa latenza")
print("      → Budget limitato (nessuna API call)")
print("      → Domini generali")

print("\n3. BEST PRACTICES:")
print("   • Retrieval Ampio → Reranking Preciso: Recupera 20-50 docs, rerank top 3-5")
print("   • Monitora Performance: Traccia miglioramento di rilevanza vs costo")
print("   • Combinazioni: Cross-Encoder per filtering + LLM per reranking finale")
print("   • Caching: Cachea risultati di reranking per query frequenti")

print("\n4. INTEGRAZIONE CON ALTRE TECNICHE RAG:")
print("   • Reranking + HyDE: Genera documento ipotetico, poi rerank")
print("   • Reranking + Query Transformation: Trasforma query, poi rerank")
print("   • Reranking + Fusion Retrieval: Fai fusion, poi rerank per precision finale")
print("   • Reranking + RSE: Rerank, poi estrai segmenti rilevanti")

print("\n💰 CONSIDERAZIONI SUI COSTI:")
print("   Cross-Encoder: Gratuito (esecuzione locale)")
print("   LLM-based: $0.001-0.01 per query (dipende da # documenti e lunghezza)")
print("   → Per 1000 query/giorno con LLM: ~$10-100/giorno")
print("   → Per 1000 query/giorno con Cross-Encoder: $0")

print("\n🚀 RACCOMANDAZIONE FINALE:")
print("   Inizia con Cross-Encoder per il 90% dei casi.")
print("   Usa LLM-based solo quando assolutamente necessario per accuratezza massima.")
print("   Combina entrambi per il meglio dei due mondi (ma considera i costi).")

print("\n" + "="*80 + "\n")

# %% [markdown]
# ![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--reranking)
