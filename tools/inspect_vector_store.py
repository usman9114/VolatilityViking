
import chromadb
import os
import json
from dotenv import load_dotenv

load_dotenv('profiles/usman/.env')

def inspect_db():
    try:
        client = chromadb.PersistentClient(path="data/chroma_db")
        
        # We need the embedding function to get the collection, even if we just peek?
        # Actually peeking doesn't require embedding function strictly if we don't query by text.
        # But let's try to get collection without it first or with a dummy if needed.
        # Ideally we use the same setup.
        
        from chromadb.utils import embedding_functions
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        
        collection = client.get_collection(name="global_patterns", embedding_function=openai_ef)
        
        print(f"Count: {collection.count()}")
        
        # Peek at 5 items
        res = collection.peek(limit=5)
        
        print("\n--- METADATA INSPECTION ---")
        for i, meta in enumerate(res['metadatas']):
            print(f"Item {i}: {meta}")
            
        print("\n--- DOCUMENT TEXT INSPECTION ---")
        for i, doc in enumerate(res['documents']):
            print(f"Item {i}: {doc}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_db()
