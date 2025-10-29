import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure
import os
from dotenv import load_dotenv

# Loads variables from .env file if they are not already set in the environment
load_dotenv()

# Best practice: store your credentials in environment variables
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

if not weaviate_url or not weaviate_api_key:
    raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY must be set in .env file or environment variables")

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,  # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(weaviate_api_key),  # Replace with your Weaviate Cloud key
    headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")
    }
)

# Check if collection already exists
if client.collections.exists("Question"):
    print("Collection 'Question' already exists. Deleting it first...")
    client.collections.delete("Question")

# Create the collection
questions = client.collections.create(
    name="Question",
    vector_config=Configure.Vectors.text2vec_weaviate(),  # Configure the Weaviate Embeddings integration
)

print("Collection 'Question' created successfully!")

client.close()  # Free up resources