import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams

QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

COLLECTION_NAME = "hs_aalen_hybrid"

print(f"📦 Creating/Updating collection '{COLLECTION_NAME}'...")

try:
    client.delete_collection(COLLECTION_NAME)
    print(f"🗑️ Existing collection '{COLLECTION_NAME}' deleted.")
except:
    pass

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": VectorParams(size=384, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(
            index=SparseIndexParams(
                on_disk=False,
            )
        )
    }
)

print(f"✅ Collection '{COLLECTION_NAME}' created with Dense + Sparse support.")
