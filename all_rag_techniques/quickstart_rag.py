import os
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from dotenv import load_dotenv
import google.generativeai as genai

# Loads variables from .env file if they are not already set in the environment
load_dotenv()

# Best practice: store your credentials in environment variables
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not weaviate_url or not weaviate_api_key:
    raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY must be set in .env file or environment variables")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY must be set in .env file or environment variables")

# Configure Gemini
genai.configure(api_key=gemini_api_key)

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
)

questions = client.collections.use("Question")

# Perform vector search
response = questions.query.near_text(
    query="biology",
    limit=2,
    return_metadata=MetadataQuery(distance=True)
)

print(f"Found {len(response.objects)} results for query 'biology':\n")

# Collect the retrieved context
context_items = []
for obj in response.objects:
    context_items.append(f"Q: {obj.properties['question']}\nA: {obj.properties['answer']}\nCategory: {obj.properties['category']}")
    print(f"Question: {obj.properties['question']}")
    print(f"Answer: {obj.properties['answer']}")
    print(f"Category: {obj.properties['category']}")
    print("-" * 50)

# Generate response using Gemini
context = "\n\n".join(context_items)
prompt = f"""Based on these facts from a quiz game:

{context}

Write a tweet with emojis about these facts."""

print("\n" + "=" * 50)
print("Generating response with Gemini 2.5 Flash...")
print("=" * 50 + "\n")

model = genai.GenerativeModel('gemini-2.0-flash-exp')
gemini_response = model.generate_content(prompt)

print("Generated Tweet:")
print(gemini_response.text)

client.close()  # Free up resources
