# %% [markdown]
# # HyDE - Hypothetical Document Embeddings (Embeddings di Documenti Ipotetici)
# 
# ## Panoramica
# 
# **TECNICA INNOVATIVA**: HyDE implementa un sistema di retrieval di documenti
# trasformando le query in documenti ipotetici. A differenza dei metodi tradizionali
# che soffrono del gap semantico tra query brevi e documenti lunghi, HyDE espande
# la query in un documento completo, migliorando potenzialmente la rilevanza del
# retrieval rendendo la rappresentazione della query più simile a quella dei
# documenti nello spazio vettoriale.
# 
# ## Motivazione
# 
# I metodi di retrieval tradizionali spesso faticano con il gap semantico tra
# query brevi e documenti più lunghi e dettagliati. HyDE affronta questo problema
# espandendo la query in un documento ipotetico completo, migliorando potenzialmente
# la rilevanza del retrieval facendo sì che la rappresentazione della query sia
# più simile alle rappresentazioni dei documenti nello spazio vettoriale.
# 
# ## Componenti Chiave
# 
# 1. Caricamento e chunking del testo
# 2. Creazione vector store usando Chroma e Google Embeddings
# 3. Language model per generare documenti ipotetici
# 4. Classe custom HyDERetriever che implementa la tecnica HyDE
# 
# ## Dettagli del Metodo
# 
# ### Preprocessing del Documento e Creazione Vector Store
# 
# 1. Il file di testo viene processato e suddiviso in chunk
# 2. Un vector store Chroma viene creato usando Google Embeddings per
#    ricerca di similarità efficiente
# 
# ### Generazione Documento Ipotetico
# 
# 1. Un language model (Gemini) viene usato per generare un documento ipotetico
#    che risponde alla query data
# 2. La generazione è guidata da un prompt template che assicura che il documento
#    ipotetico sia dettagliato e corrisponda alla dimensione dei chunk usati
#    nel vector store
# 
# ### Processo di Retrieval
# 
# La classe `HyDERetriever` implementa i seguenti passaggi:
# 
# 1. Genera un documento ipotetico dalla query usando il language model
# 2. Usa il documento ipotetico come query di ricerca nel vector store
# 3. Recupera i documenti più simili a questo documento ipotetico
# 
# ## Caratteristiche Chiave
# 
# 1. **Query Expansion**: Trasforma query brevi in documenti ipotetici dettagliati
# 2. **Configurazione Flessibile**: Permette aggiustamento di chunk size, overlap
#    e numero di documenti recuperati
# 3. **Integrazione con Gemini**: Usa Gemini per generazione di documenti ipotetici
#    e Google Embeddings per rappresentazione vettoriale
# 
# ## Vantaggi di Questo Approccio
# 
# 1. **Rilevanza Migliorata**: Espandendo le query in documenti completi, HyDE
#    può potenzialmente catturare match più sfumati e rilevanti
# 2. **Gestione Query Complesse**: Particolarmente utile per query complesse o
#    multi-sfaccettate che potrebbero essere difficili da matchare direttamente
# 3. **Adattabilità**: La generazione di documenti ipotetici può adattarsi a
#    diversi tipi di query e domini di documenti
# 4. **Potenziale per Migliore Comprensione del Contesto**: La query espansa
#    potrebbe catturare meglio il contesto e l'intento dietro la domanda originale
# 
# ## Dettagli di Implementazione
# 
# 1. Usa il modello Gemini di Google per generazione di documenti ipotetici
# 2. Usa Chroma per ricerca di similarità efficiente nello spazio vettoriale
# 3. Permette facile visualizzazione sia del documento ipotetico che dei
#    risultati recuperati
# 
# ## Conclusione
# 
# Hypothetical Document Embedding (HyDE) rappresenta un approccio innovativo al
# retrieval di documenti, affrontando il gap semantico tra query e documenti.
# Sfruttando language model avanzati per espandere le query in documenti ipotetici,
# HyDE ha il potenziale di migliorare significativamente la rilevanza del retrieval,
# specialmente per query complesse o sfumate. Questa tecnica potrebbe essere
# particolarmente preziosa in domini dove comprendere l'intento e il contesto della
# query è cruciale, come ricerca legale, revisione di letteratura accademica, o
# sistemi avanzati di information retrieval.

# %% [markdown]
# <div style="text-align: center;">
# 
# <img src="../images/HyDe.svg" alt="HyDe" style="width:40%; height:auto;">
# </div>

# %% [markdown]
# <div style="text-align: center;">
# 
# <img src="../images/hyde-advantages.svg" alt="HyDe" style="width:100%; height:auto;">
# </div>

# %% [markdown]
# # Imports e Setup

# %%
import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Setup percorso al modulo parent per importare evaluation (se necessario)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Carica variabili d'ambiente
load_dotenv()

# Configura API key Gemini
if not os.getenv('GEMINI_API_KEY'):
    raise ValueError("GEMINI_API_KEY non trovata nel file .env")
os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')

# %% [markdown]
# ### Configurazione Documento e Modelli

# %%
# Configurazione documento e modelli
script_dir = os.path.dirname(os.path.abspath(__file__))
default_data_dir = os.path.join(os.path.dirname(script_dir), 'data')
PATH = os.path.join(default_data_dir, "cambiamento_climatico.txt")

# Modelli Gemini
LANGUAGE_MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025"
EMBEDDING_MODEL_NAME = "models/embedding-001"

# Parametri di chunking (allineati con HyPE)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# %% [markdown]
# ### Definizione della Classe HyDERetriever

# %%
class HyDERetriever:
    """
    Implementa il retrieval HyDE (Hypothetical Document Embeddings).
    
    TECNICA HyDE:
    Invece di fare embedding diretto della query dell'utente, HyDE genera
    un "documento ipotetico" che risponde alla query, poi usa l'embedding
    di questo documento per il retrieval.
    
    VANTAGGI:
    - Elimina il mismatch di stile query-documento
    - La query è breve e interrogativa, il documento è descrittivo
    - Il documento ipotetico ha lo stesso stile dei chunk nel vector store
    - Migliora significativamente la precisione del retrieval
    
    DIFFERENZA DA RAG TRADIZIONALE:
    - RAG tradizionale: embedding(query) → retrieval
    - HyDE: query → genera_documento_ipotetico → embedding(documento) → retrieval
    """
    
    def __init__(self, file_path, chunk_size=500, chunk_overlap=100):
        """
        Inizializza il retriever HyDE.
        
        Args:
            file_path: Percorso al file di testo da indicizzare
            chunk_size: Dimensione dei chunk
            chunk_overlap: Overlap tra chunk consecutivi
        """
        # Modelli Gemini
        self.llm = ChatGoogleGenerativeAI(
            model=LANGUAGE_MODEL_NAME,
            temperature=0,
            max_tokens=4000
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Crea vector store con chunking normale
        self.vectorstore = self._encode_text_file(file_path)
        
        # Prompt per generazione documento ipotetico (in italiano)
        self.hyde_prompt = PromptTemplate(
            input_variables=["query", "chunk_size"],
            template="""Data la domanda '{query}', genera un documento ipotetico dettagliato che risponde direttamente a questa domanda.

Il documento deve:
- Essere scritto in italiano
- Essere approfondito e ricco di informazioni
- Avere esattamente {chunk_size} caratteri
- Usare uno stile descrittivo e informativo (non interrogativo)
- Contenere informazioni concrete e specifiche

Documento ipotetico:""",
        )
        self.hyde_chain = self.hyde_prompt | self.llm

    def _encode_text_file(self, file_path):
        """
        Carica un file di testo e crea un vector store con chunking NORMALE.
        
        NOTA: A differenza di HyPE, qui l'indicizzazione è standard.
        La generazione ipotetica avviene solo a runtime per le query.
        """
        print(f"\n📂 Caricamento documento: {os.path.basename(file_path)}")
        
        # Carica il file di testo
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"📄 Documento caricato: {len(text)} caratteri")
        
        # Suddividi in chunk
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        documents = [Document(page_content=text, metadata={"source": file_path})]
        chunks = text_splitter.split_documents(documents)
        
        print(f"✂️  Creati {len(chunks)} chunk")
        print(f"   Dimensione media: {sum(len(c.page_content) for c in chunks)/len(chunks):.0f} caratteri")
        
        # Crea vector store Chroma con embeddings normali
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name="hyde_chunks"
        )
        
        print("✅ Vector store creato con indicizzazione standard")
        return vectorstore

    def generate_hypothetical_document(self, query):
        """
        Genera un documento ipotetico che risponde alla query.
        
        CUORE DELLA TECNICA HyDE:
        Trasforma una query breve in un documento dettagliato nello stesso
        stile dei documenti nel vector store.
        
        Args:
            query: Query dell'utente
            
        Returns:
            str: Documento ipotetico generato
        """
        input_variables = {"query": query, "chunk_size": self.chunk_size}
        hypothetical_doc = self.hyde_chain.invoke(input_variables).content
        return hypothetical_doc

    def retrieve(self, query, k=3):
        """
        Esegue il retrieval HyDE completo.
        
        PIPELINE:
        1. Genera documento ipotetico dalla query
        2. Crea embedding del documento ipotetico
        3. Cerca chunk simili nel vector store
        4. Ritorna chunk + documento ipotetico
        
        Args:
            query: Query dell'utente
            k: Numero di chunk da recuperare
            
        Returns:
            tuple: (chunk_recuperati, documento_ipotetico)
        """
        # Genera documento ipotetico
        hypothetical_doc = self.generate_hypothetical_document(query)
        
        # Usa il documento ipotetico per il retrieval
        similar_docs = self.vectorstore.similarity_search(hypothetical_doc, k=k)
        
        return similar_docs, hypothetical_doc

# %% [markdown]
# ### Fase 1: Creazione del Retriever HyDE

# %%
print("\n" + "="*80)
print("🚀 FASE 1: Creazione HyDE Retriever")
print("="*80)

retriever = HyDERetriever(PATH, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

print("\n✅ Retriever HyDE pronto!")

# %% [markdown]
# ### Fase 2: Test con Prima Query (Sbiancamento Coralli)

# %%
print("\n" + "="*80)
print("🧪 FASE 2: Test con Query di Esempio")
print("="*80)

# Query 1: Causa diretta dello sbiancamento dei coralli
test_query = "Qual è la causa diretta dello sbiancamento dei coralli?"
print(f"\n📋 Query: {test_query}")
print("\n⏳ Generazione documento ipotetico...")

results, hypothetical_doc = retriever.retrieve(test_query, k=3)

# %% [markdown]
# ### Visualizzazione Documento Ipotetico (Caratteristica Chiave di HyDE)

# %%
print("\n" + "="*80)
print("📄 DOCUMENTO IPOTETICO GENERATO (caratteristica chiave di HyDE)")
print("="*80)
print("\n🤖 Gemini ha generato il seguente documento per rispondere alla query:")
print("\n" + "-"*80)
print(hypothetical_doc)
print("-"*80)
print(f"\n📊 Lunghezza: {len(hypothetical_doc)} caratteri")

# %% [markdown]
# ### Chunk Recuperati Basati sul Documento Ipotetico

# %%
print("\n" + "="*80)
print("📚 CHUNK RECUPERATI (basati sul documento ipotetico)")
print("="*80)

for i, doc in enumerate(results, 1):
    print(f"\n📄 Chunk {i}:")
    print("-"*80)
    print(doc.page_content)
    print("-"*80)

# %% [markdown]
# ### Fase 3: Secondo Test (Tecnologie Innovative)

# %%
print("\n" + "="*80)
print("🧪 FASE 3: Secondo Test - Query su Tecnologie")
print("="*80)

test_query_2 = "Quali sono le tecnologie innovative per combattere il cambiamento climatico?"
print(f"\n📋 Query: {test_query_2}")
print("\n⏳ Generazione documento ipotetico...")

results_2, hypothetical_doc_2 = retriever.retrieve(test_query_2, k=3)

# %% [markdown]
# ### Visualizzazione Secondo Documento Ipotetico

# %%
print("\n" + "="*80)
print("📄 DOCUMENTO IPOTETICO GENERATO")
print("="*80)
print("\n🤖 Gemini ha generato:")
print("\n" + "-"*80)
print(hypothetical_doc_2)
print("-"*80)
print(f"\n📊 Lunghezza: {len(hypothetical_doc_2)} caratteri")

# %% [markdown]
# ### Chunk Recuperati per la Seconda Query

# %%
print("\n" + "="*80)
print("📚 CHUNK RECUPERATI")
print("="*80)

for i, doc in enumerate(results_2, 1):
    print(f"\n📄 Chunk {i}:")
    print("-"*80)
    print(doc.page_content)
    print("-"*80)

# %% [markdown]
# ### Riepilogo Finale e Lezioni Chiave

# %%
print("\n" + "="*80)
print("✅ DEMO COMPLETATA: HyDE - Hypothetical Document Embeddings")
print("="*80)
print("\n🎯 LEZIONI CHIAVE:")
print("   1. HyDE genera documenti ipotetici A RUNTIME per ogni query")
print("      → Trasforma query brevi in documenti dettagliati")
print("   2. Il retrieval usa l'embedding del documento ipotetico")
print("      → Elimina il mismatch query-documento")
print("   3. L'indicizzazione è standard (embeddings normali dei chunk)")
print("      → La 'magia' avviene solo al momento della query")
print("   4. Trade-off: precisione aumentata vs latenza query")
print("      → Ogni query richiede una chiamata LLM aggiuntiva")
print("\n💡 QUANDO USARE HyDE:")
print("   ✅ Query complesse che richiedono risposte dettagliate")
print("   ✅ Domini tecnici dove espandere la query aiuta")
print("   ✅ Quando la precisione è prioritaria rispetto alla latenza")
print("   ✅ Documenti lunghi e dettagliati nel vector store")
print("\n⚠️  QUANDO HyDE HA MENO IMPATTO:")
print("   - Query già dettagliate o documenti brevi")
print("   - Necessità di latenza minima (HyDE aggiunge overhead)")
print("   - Budget limitato per chiamate LLM (costa ad ogni query)")
print("\n" + "="*80 + "\n")

# %% [markdown]
# ## 🎓 Analisi Tecnica: HyDE vs HyPE vs RAG Tradizionale
#
# ### Confronto delle Tre Tecniche
#
# **RAG Tradizionale:**
# ```
# Indicizzazione: chunks → embeddings(chunks) → vector store
# Runtime: query → embedding(query) → similarity search → retrieval
# ```
# - Problema: Mismatch di stile query-documento
# - Costo runtime: solo 1 embedding call
#
# **HyDE (questo script):**
# ```
# Indicizzazione: chunks → embeddings(chunks) → vector store
# Runtime: query → genera_documento_ipotetico → embedding(documento) → retrieval
# ```
# - Soluzione: Espande la query in documento ipotetico (stesso stile dei chunk)
# - Costo runtime: 1 LLM call + 1 embedding call
#
# **HyPE (script precedente):**
# ```
# Indicizzazione: chunks → genera_domande → embeddings(domande) → vector store
# Runtime: query → embedding(query) → similarity search → retrieval
# ```
# - Soluzione: Pre-genera domande ipotetiche (matching domanda-domanda)
# - Costo runtime: solo 1 embedding call
#
# ### Trade-offs Chiave
#
# | Aspetto | RAG Tradizionale | HyDE | HyPE |
# |---------|-----------------|------|------|
# | Costo Indicizzazione | Basso | Basso | Alto |
# | Costo Runtime | Basso | Medio-Alto | Basso |
# | Latenza Query | Bassa | Alta | Bassa |
# | Precisione | Baseline | Alta | Molto Alta |
# | Flessibilità | Media | Alta | Media |
#
# ### Quando Usare Cosa?
#
# **Usa RAG Tradizionale se:**
# - Budget limitato
# - Latenza critica
# - Query e documenti già ben allineati
#
# **Usa HyDE se:**
# - Query complesse e articolate
# - Precisione prioritaria su latenza
# - Budget per chiamate LLM a runtime
# - Necessità di adattarsi dinamicamente a nuovi tipi di query
#
# **Usa HyPE se:**
# - Pattern di query prevedibili
# - Latenza critica (come RAG tradizionale)
# - Budget per setup iniziale costoso
# - Necessità di massima precisione con bassa latenza
#
# ### Best Practices per HyDE
#
# 1. **Ottimizza il Prompt di Generazione**
#    - Specifica lunghezza target del documento
#    - Richiedi stile descrittivo (non interrogativo)
#    - Includi esempi se necessario
#
# 2. **Caching Intelligente**
#    - Cachea documenti ipotetici per query frequenti
#    - Riduci il numero di chiamate LLM ripetitive
#
# 3. **Monitora Qualità e Costi**
#    - Verifica che i documenti generati siano di alta qualità
#    - Traccia il costo delle chiamate LLM
#    - Considera hybrid approach per query semplici
#
# 4. **Combina con Altre Tecniche**
#    - HyDE + Reranking: Massima precisione
#    - HyDE + Caching: Bilancia costo/performance
#    - HyDE selettivo: Usa solo per query complesse

# %% [markdown]
# ![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--hyde-hypothetical-document-embedding)
