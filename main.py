from ingestion.load_text import load_text
from preprocessing.clean_text import clean_text
from chunking.chunk_text import chunk_text
from embeddings.embed_chunks import embed_chunks
from vector_store.faiss_index import create_faiss_index
from retrieval.retrieve import retrieve_chunks
from llm.generate_answer import generate_answer
from sentence_transformers import SentenceTransformer

text = load_text('data/sample.txt')

cleaned = clean_text(text)
chunks = chunk_text(cleaned)

embeddings = embed_chunks(chunks)
index = create_faiss_index(embeddings)

model = SentenceTransformer("all-MiniLM-L6-v2")

query = input("Ask a Question: ")

retrieved_chunks = retrieve_chunks(
    query,
    model,
    index,
    chunks,
    k=3
)

context = "\n".join(retrieved_chunks)

answer = generate_answer(context, query)

print("\Answer:\n")
print(answer)