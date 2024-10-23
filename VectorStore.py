import numpy as np
import faiss
from Embeddings import embeddings

embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)

embeddings_np = np.array(embeddings, dtype=np.float32)
index.add(embeddings_np)

faiss.write_index(index, 'course_embeddings.index')

print("FAISS index saved successfully to 'course_embeddings.index'")
