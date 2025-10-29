"""
Test Diagnostico Gemini Embeddings API

Questo script testa la configurazione dell'API key Gemini con un numero
minimo di richieste per diagnosticare problemi di quota o configurazione.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set the Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')

print("="*70)
print("TEST DIAGNOSTICO GEMINI EMBEDDINGS API")
print("="*70)

# Verify API key is loaded
api_key = os.getenv('GEMINI_API_KEY')
if api_key:
    print(f"✓ API Key trovata: {api_key[:10]}...{api_key[-4:]}")
else:
    print("✗ ERRORE: API Key non trovata!")
    print("  Verifica che GEMINI_API_KEY sia impostata nel file .env")
    exit(1)

print("\n" + "-"*70)
print("Test 1: Importazione librerie LangChain")
print("-"*70)

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    print("✓ Import GoogleGenerativeAIEmbeddings riuscito")
except Exception as e:
    print(f"✗ Errore import: {e}")
    exit(1)

print("\n" + "-"*70)
print("Test 2: Inizializzazione modello embeddings")
print("-"*70)

try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("✓ Modello inizializzato correttamente")
except Exception as e:
    print(f"✗ Errore inizializzazione: {e}")
    exit(1)

print("\n" + "-"*70)
print("Test 3: Embedding di UN SINGOLO testo (query)")
print("-"*70)

try:
    test_text = "This is a simple test."
    print(f"   Testo: '{test_text}'")
    embedding = embeddings.embed_query(test_text)
    print(f"✓ Embedding generato con successo!")
    print(f"   Dimensione vettore: {len(embedding)}")
    print(f"   Primi 5 valori: {embedding[:5]}")
except Exception as e:
    print(f"✗ ERRORE durante embed_query:")
    print(f"   Tipo: {type(e).__name__}")
    print(f"   Messaggio: {str(e)[:200]}")
    if "429" in str(e) or "quota" in str(e).lower():
        print("\n⚠️  PROBLEMA DI QUOTA/BILLING RILEVATO!")
        print("   Possibili cause:")
        print("   1. API key non ha accesso a embeddings (free tier limitato)")
        print("   2. Billing non attivo sull'account Google Cloud")
        print("   3. Quota giornaliera superata")
        print("\n   Soluzioni:")
        print("   - Verifica su: https://ai.google.dev/")
        print("   - Controlla: https://ai.dev/usage?tab=rate-limit")
    exit(1)

print("\n" + "-"*70)
print("Test 4: Embedding di DUE testi (batch)")
print("-"*70)

try:
    test_texts = [
        "First test document.",
        "Second test document."
    ]
    print(f"   Numero testi: {len(test_texts)}")
    embeddings_batch = embeddings.embed_documents(test_texts)
    print(f"✓ Batch embedding generato con successo!")
    print(f"   Numero vettori: {len(embeddings_batch)}")
    print(f"   Dimensione ciascun vettore: {len(embeddings_batch[0])}")
except Exception as e:
    print(f"✗ ERRORE durante embed_documents:")
    print(f"   Tipo: {type(e).__name__}")
    print(f"   Messaggio: {str(e)[:200]}")
    if "429" in str(e) or "quota" in str(e).lower():
        print("\n⚠️  PROBLEMA DI QUOTA/BILLING RILEVATO!")
        print("   Il singolo embedding ha funzionato ma il batch no.")
        print("   Questo suggerisce limiti molto bassi sulla quota.")
    exit(1)

print("\n" + "="*70)
print("✅ TUTTI I TEST SUPERATI!")
print("="*70)
print("\nLa tua API key Gemini è configurata correttamente per embeddings.")
print("Il problema nel tuo script principale è dovuto al numero elevato")
print("di chunks (4155 chunks = 4155 richieste di embedding).")
print("\nSoluzioni consigliate:")
print("  1. Riduci chunk_size per avere meno chunks")
print("  2. Riduci num_eval_questions per testare")
print("  3. Limita il numero di documenti da processare")
print("  4. Verifica/aumenta la quota su https://ai.google.dev/")

