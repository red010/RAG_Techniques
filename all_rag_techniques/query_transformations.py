# %% [markdown]
# # Query Transformations per RAG Avanzato (Trasformazioni di Query)
# 
# ## Panoramica
# 
# **TECNICHE FONDAMENTALI**: Questo script implementa tre tecniche di trasformazione
# delle query per migliorare il retrieval nei sistemi RAG:
# 
# 1. **Query Rewriting** (Riscrittura Query)
# 2. **Step-back Prompting** (Generalizzazione)
# 3. **Sub-query Decomposition** (Decomposizione)
# 
# Ogni tecnica modifica o espande la query originale per migliorare la rilevanza
# e la completezza delle informazioni recuperate.
# 
# ## Motivazione
# 
# I sistemi RAG spesso hanno difficoltà a recuperare le informazioni più rilevanti,
# specialmente con query complesse o ambigue. Le query transformation affrontano
# questo problema riformulando le query per:
# 
# - **Migliorare il matching** con documenti rilevanti
# - **Recuperare contesto** più ampio e completo
# - **Suddividere complessità** in parti gestibili
# 
# ## Le Tre Tecniche in Dettaglio
# 
# ### 1. Query Rewriting (Riscrittura)
# 
# **SCOPO**: Rendere la query più specifica e dettagliata, aumentando la probabilità
# di recuperare informazioni rilevanti.
# 
# **QUANDO USARLA**:
# - Query vaghe o generiche
# - Necessità di specificare aspetti particolari
# - Migliorare la precisione del retrieval
# 
# **ESEMPIO**:
# ```
# Originale: "Impatti del clima"
# Rewritten: "Quali sono gli impatti specifici del cambiamento climatico
#             sulla biodiversità, temperature globali e eventi estremi?"
# ```
# 
# ### 2. Step-back Prompting (Generalizzazione)
# 
# **SCOPO**: Generare una query più ampia e generale per recuperare informazioni
# di contesto e background rilevanti.
# 
# **QUANDO USARLA**:
# - Query troppo specifiche che potrebbero perdere contesto
# - Necessità di comprendere il quadro generale
# - Prima fase di un retrieval a due livelli
# 
# **ESEMPIO**:
# ```
# Originale: "Come l'acidificazione degli oceani influenza i coralli?"
# Step-back: "Quali sono gli effetti generali del cambiamento climatico
#             sugli ecosistemi marini?"
# ```
# 
# ### 3. Sub-query Decomposition (Decomposizione)
# 
# **SCOPO**: Scomporre query complesse in 2-4 sotto-query più semplici per
# un retrieval più completo e mirato.
# 
# **QUANDO USARLA**:
# - Query multi-aspetto o multi-dominio
# - Necessità di coprire diverse sfaccettature di un argomento
# - Retrieval parallelo per performance
# 
# **ESEMPIO**:
# ```
# Originale: "Impatti e soluzioni del cambiamento climatico"
# Sub-queries:
#   1. "Quali sono gli impatti del cambiamento climatico?"
#   2. "Quali sono le soluzioni tecnologiche al cambiamento climatico?"
#   3. "Quali sono le politiche per affrontare il cambiamento climatico?"
# ```
# 
# ## Vantaggi di Queste Tecniche
# 
# 1. **Rilevanza Migliorata**: Query rewriting recupera informazioni più specifiche
# 2. **Contesto Migliore**: Step-back prompting fornisce background essenziale
# 3. **Completezza**: Decomposition copre aspetti multipli di query complesse
# 4. **Flessibilità**: Ogni tecnica può essere usata singolarmente o in combinazione
# 
# ## Implementazione
# 
# - Tutte le tecniche usano Gemini per la trasformazione
# - Template di prompt customizzati guidano il modello
# - Funzioni separate permettono facile integrazione in sistemi RAG esistenti
# 
# ## Conclusione
# 
# Le query transformation sono strumenti potenti per migliorare il retrieval RAG.
# Riformulando le query in modi diversi, possono significativamente migliorare
# rilevanza, contesto e completezza delle informazioni recuperate. Sono
# particolarmente preziose in domini dove le query possono essere complesse o
# multi-sfaccettate, come ricerca scientifica, analisi legale, o fact-finding.

# %% [markdown]
# # Setup e Imports

# %%
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()
if not os.getenv('GEMINI_API_KEY'):
    raise ValueError("GEMINI_API_KEY non trovata nel file .env")
os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')

# Modello Gemini
LANGUAGE_MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025"

# %% [markdown]
# ### Query di Test: Complessa e Multi-Aspetto
# 
# Usiamo una query complessa sul cambiamento climatico che beneficia
# da tutte e tre le tecniche di trasformazione.

# %%
# Query complessa che richiede trasformazione per retrieval ottimale
ORIGINAL_QUERY = """Come influisce il cambiamento climatico sull'agricoltura 
e quali sono le strategie di adattamento più efficaci?"""

print("\n" + "="*80)
print("📋 QUERY ORIGINALE DA TRASFORMARE")
print("="*80)
print(f"\n{ORIGINAL_QUERY}")
print("\n💡 Questa è una query complessa che:")
print("   - È multi-aspetto (impatti + soluzioni)")
print("   - Potrebbe essere troppo ampia (beneficia da rewriting)")
print("   - Ha bisogno di contesto (beneficia da step-back)")
print("   - Copre temi diversi (beneficia da decomposition)")

# %% [markdown]
# ### Tecnica 1: Query Rewriting (Riscrittura Query)
# 
# Riformula la query per renderla più specifica e dettagliata.

# %%
print("\n" + "="*80)
print("🔧 TECNICA 1: QUERY REWRITING (Riscrittura)")
print("="*80)

rewrite_llm = ChatGoogleGenerativeAI(
    model=LANGUAGE_MODEL_NAME,
    temperature=0
)

# Template per query rewriting in italiano
query_rewrite_template = """Sei un assistente AI specializzato nel riformulare query per migliorare il retrieval in sistemi RAG.
Data una query originale, riscrivila per renderla più specifica, dettagliata e più probabile nel recuperare informazioni rilevanti.

ISTRUZIONI:
- Mantieni lo stesso intento della query originale
- Aggiungi specificità e dettagli
- Usa termini tecnici appropriati quando rilevanti
- Mantieni la query in italiano

Query originale: {original_query}

Query riscritta:"""

query_rewrite_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=query_rewrite_template
)

# Crea chain per query rewriting
query_rewriter = query_rewrite_prompt | rewrite_llm

def rewrite_query(original_query: str) -> str:
    """
    Riscrive la query originale per migliorare il retrieval.
    
    TECNICA: Query Rewriting
    Trasforma query vaghe o generiche in query specifiche e dettagliate,
    aumentando la probabilità di recuperare documenti rilevanti.
    
    Args:
        original_query: La query originale dell'utente
        
    Returns:
        La query riscritta più specifica
    """
    response = query_rewriter.invoke({"original_query": original_query})
    return response.content

# %% [markdown]
# ### Tecnica 2: Step-back Prompting (Generalizzazione)
# 
# Genera una query più ampia per recuperare contesto di background.

# %%
print("\n" + "="*80)
print("🔙 TECNICA 2: STEP-BACK PROMPTING (Generalizzazione)")
print("="*80)

step_back_llm = ChatGoogleGenerativeAI(
    model=LANGUAGE_MODEL_NAME,
    temperature=0
)

# Template per step-back prompting in italiano
step_back_template = """Sei un assistente AI specializzato nel generare query più ampie e generali per migliorare il retrieval di contesto in sistemi RAG.
Data una query originale (che potrebbe essere molto specifica), genera una query "step-back" più generale che possa aiutare a recuperare informazioni di background rilevanti.

ISTRUZIONI:
- Genera una query più astratta e generale
- Mantieni la rilevanza all'argomento originale
- La query step-back dovrebbe coprire il contesto più ampio
- Mantieni la query in italiano

Query originale: {original_query}

Query step-back (più generale):"""

step_back_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=step_back_template
)

# Crea chain per step-back prompting
step_back_chain = step_back_prompt | step_back_llm

def generate_step_back_query(original_query: str) -> str:
    """
    Genera una query step-back più generale per recuperare contesto.
    
    TECNICA: Step-back Prompting
    Trasforma query specifiche in query generali che recuperano informazioni
    di background e contesto essenziali per comprendere meglio l'argomento.
    
    Args:
        original_query: La query originale dell'utente
        
    Returns:
        La query step-back più generale
    """
    response = step_back_chain.invoke({"original_query": original_query})
    return response.content

# %% [markdown]
# ### Tecnica 3: Sub-query Decomposition (Decomposizione)
# 
# Scompone query complesse in sotto-query più semplici.

# %%
print("\n" + "="*80)
print("🔀 TECNICA 3: SUB-QUERY DECOMPOSITION (Decomposizione)")
print("="*80)

sub_query_llm = ChatGoogleGenerativeAI(
    model=LANGUAGE_MODEL_NAME,
    temperature=0
)

# Template per sub-query decomposition in italiano
subquery_decomposition_template = """Sei un assistente AI specializzato nel scomporre query complesse in sotto-query più semplici per sistemi RAG.
Data una query originale complessa, scomponila in 2-4 sotto-query più semplici che, se risposte insieme, fornirebbero una risposta completa alla query originale.

ISTRUZIONI:
- Genera 2-4 sotto-query (non di più)
- Ogni sotto-query deve essere autonoma e chiara
- Insieme, le sotto-query devono coprire tutti gli aspetti della query originale
- Numera le sotto-query (1., 2., 3., ecc.)
- Mantieni le sotto-query in italiano

Query originale: {original_query}

Sotto-query:"""

subquery_decomposition_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=subquery_decomposition_template
)

# Crea chain per sub-query decomposition
subquery_decomposer_chain = subquery_decomposition_prompt | sub_query_llm

def decompose_query(original_query: str) -> list:
    """
    Scompone la query originale in sotto-query più semplici.
    
    TECNICA: Sub-query Decomposition
    Trasforma query complesse multi-aspetto in un set di sotto-query semplici,
    permettendo retrieval parallelo e più completo.
    
    Args:
        original_query: La query complessa originale
        
    Returns:
        Lista di sotto-query più semplici
    """
    response = subquery_decomposer_chain.invoke({"original_query": original_query}).content
    # Estrai le sotto-query filtrandole
    sub_queries = [q.strip() for q in response.split('\n') if q.strip() and any(c.isdigit() for c in q[:3])]
    return sub_queries

# %% [markdown]
# ### Dimostrazione ed Analisi Comparativa

# %%
print("\n" + "="*80)
print("🔍 FASE 1: Applicazione delle 3 Tecniche")
print("="*80)

# Applica tutte e tre le tecniche alla stessa query
print("\n⏳ Applicazione Query Rewriting...")
rewritten_query = rewrite_query(ORIGINAL_QUERY)

print("⏳ Applicazione Step-back Prompting...")
step_back_query = generate_step_back_query(ORIGINAL_QUERY)

print("⏳ Applicazione Sub-query Decomposition...")
sub_queries = decompose_query(ORIGINAL_QUERY)

print("\n✅ Tutte le trasformazioni completate!")

# %% [markdown]
# ### Visualizzazione Risultati

# %%
print("\n" + "="*80)
print("📊 FASE 2: Risultati delle Trasformazioni")
print("="*80)

print("\n" + "-"*80)
print("📌 QUERY ORIGINALE:")
print("-"*80)
print(f"{ORIGINAL_QUERY}")

print("\n" + "-"*80)
print("✏️  TECNICA 1: QUERY REWRITING")
print("-"*80)
print(f"{rewritten_query}")
print("\n💡 Analisi: La query è stata resa più specifica, dettagliata e strutturata.")

print("\n" + "-"*80)
print("🔙 TECNICA 2: STEP-BACK PROMPTING")
print("-"*80)
print(f"{step_back_query}")
print("\n💡 Analisi: La query è stata generalizzata per recuperare contesto più ampio.")

print("\n" + "-"*80)
print("🔀 TECNICA 3: SUB-QUERY DECOMPOSITION")
print("-"*80)
for i, sub_query in enumerate(sub_queries, 1):
    print(f"{sub_query}")
print(f"\n💡 Analisi: La query complessa è stata scomposta in {len(sub_queries)} sotto-query indipendenti.")

# %% [markdown]
# ### Analisi Comparativa Dettagliata

# %%
print("\n" + "="*80)
print("🎯 FASE 3: Analisi Comparativa e Use Cases")
print("="*80)

print("\n📊 CONFRONTO DELLE TECNICHE:")
print("\n1. QUERY REWRITING:")
print("   ✅ PRO:")
print("      - Migliora la specificità e dettaglio")
print("      - Ottimo per query vaghe o ambigue")
print("      - Aumenta precision del retrieval")
print("   ⚠️  CONTRO:")
print("      - Potrebbe restringere troppo lo scope")
print("      - Rischio di perdere informazioni correlate")

print("\n2. STEP-BACK PROMPTING:")
print("   ✅ PRO:")
print("      - Recupera contesto essenziale")
print("      - Ottimo per query troppo specifiche")
print("      - Fornisce background necessario")
print("   ⚠️  CONTRO:")
print("      - Potrebbe essere troppo generale")
print("      - Rischio di recuperare documenti non rilevanti")

print("\n3. SUB-QUERY DECOMPOSITION:")
print("   ✅ PRO:")
print("      - Copre aspetti multipli della query")
print("      - Permette retrieval parallelo")
print("      - Massima completezza della risposta")
print("   ⚠️  CONTRO:")
print("      - Costo computazionale più alto")
print("      - Necessita di aggregazione dei risultati")

print("\n💡 STRATEGIE DI COMBINAZIONE:")
print("   • Rewriting + Step-back: Per balance tra specificità e contesto")
print("   • Decomposition + Rewriting: Scomponi poi affina ogni sotto-query")
print("   • Tutte e tre: Usa rewriting per query principale, step-back per contesto,")
print("                 decomposition per aspetti multipli (massima copertura)")

print("\n🎯 QUANDO USARE COSA:")
print("\n   QUERY REWRITING:")
print("      - Query utente vaghe: 'Parlami del clima'")
print("      - Necessità di specificità: Aggiungere aspetti rilevanti")
print("      - RAG con documenti molto tecnici")
print("\n   STEP-BACK PROMPTING:")
print("      - Query utente troppo specifiche: 'Algoritmo XYZ in contesto ABC'")
print("      - Prima fase di retrieval a due livelli")
print("      - Quando il contesto è cruciale per la comprensione")
print("\n   SUB-QUERY DECOMPOSITION:")
print("      - Query multi-dominio: 'Impatti, cause e soluzioni di X'")
print("      - Necessità di copertura completa")
print("      - Sistemi RAG con retrieval parallelo")

# %% [markdown]
# ### Conclusioni e Best Practices

# %%
print("\n" + "="*80)
print("✅ DEMO COMPLETATA: Query Transformations per RAG")
print("="*80)

print("\n🎓 LEZIONI CHIAVE:")
print("   1. Le query transformation NON sono mutuamente esclusive")
print("      → Possono e dovrebbero essere combinate strategicamente")
print("   2. La scelta della tecnica dipende dal tipo di query")
print("      → Vaga → Rewriting, Specifica → Step-back, Complessa → Decomposition")
print("   3. Le trasformazioni aggiungono overhead computazionale")
print("      → Valuta il trade-off tra qualità e latenza")
print("   4. Sperimenta con i prompt per il tuo dominio specifico")
print("      → I template qui sono generici, personalizzali!")

print("\n🔬 IMPLEMENTAZIONE IN PRODUZIONE:")
print("   • Caching: Cachea le trasformazioni per query frequenti")
print("   • Thresholds: Non sempre serve trasformare (es. query già ottimali)")
print("   • Monitoraggio: Traccia quali trasformazioni migliorano i risultati")
print("   • A/B Testing: Testa diverse combinazioni di tecniche")
print("   • Fallback: Usa query originale se la trasformazione fallisce")

print("\n📚 INTEGRAZIONE CON ALTRE TECNICHE RAG:")
print("   • Query Transformations + HyDE: Trasforma poi espandi")
print("   • Query Transformations + Reranking: Trasforma poi riordina")
print("   • Query Transformations + RSE: Trasforma poi raggruppa segmenti")
print("   • Query Transformations + CCH: Trasforma poi usa header contestuali")

print("\n💰 CONSIDERAZIONI SUI COSTI:")
print("   • Query Rewriting: 1 chiamata LLM per query")
print("   • Step-back: 1 chiamata LLM per query")
print("   • Decomposition: 1 chiamata LLM + N retrieval (N = # sotto-query)")
print("   • Tutte e tre: 3 chiamate LLM per query (considera il budget!)")

print("\n" + "="*80 + "\n")

# %% [markdown]
# ![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--query-transformations)
