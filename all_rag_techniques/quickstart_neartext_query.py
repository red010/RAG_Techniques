import weaviate
from weaviate.classes.init import Auth
import os, json
from dotenv import load_dotenv

# Loads variables from .env file if they are not already set in the environment
load_dotenv()

# Best practice: store your credentials in environment variables
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

if not weaviate_url or not weaviate_api_key:
    raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY must be set in .env file or environment variables")

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,                                    # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(weaviate_api_key),             # Replace with your Weaviate Cloud key
    headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")
    }
)

questions = client.collections.use("Question")

response = questions.query.near_text(
    query="biology",
    limit=2
)

print(f"Found {len(response.objects)} results for query 'biology':\n")
for obj in response.objects:
    print(json.dumps(obj.properties, indent=2))
    print("-" * 50)

client.close()  # Free up resources