import ollama
import redis
import numpy as np
from sentence_transformers import SentenceTransformer
from redis.commands.search.query import Query
import os
import fitz
import re
import pandas as pd
import pdfplumber

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

#Vector_Dim for embedding_model1 (default) = 768, embedding_model2 = 384, embedding_model3 = 768
VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


# used to clear the redis vector store
def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")


# Create an HNSW index in Redis
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")


#Initialize embedding models
embedding_model2 = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
embedding_model3 = SentenceTransformer("all-mpnet-base-v2")

# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
   response = ollama.embeddings(model=model, prompt=text)
   return response["embedding"]

# Generate an embedding using SentenceTransformer("all-MiniLM-L6-v2")
def get_embedding2(text: str) -> list:
   response = embedding_model2.encode(text)
   return response.tolist()

# Generate an embedding using SentenceTransformer("all-mpnet-base-v2")
def get_embedding3(text: str) -> list:
   response = embedding_model3.encode(text)
   return response.tolist()




# store the embedding in Redis
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(),  # Store as byte array
        },
    )
    print(f"Stored embedding for: {chunk}")


# extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
   """Extract text from a PDF file."""
   doc = fitz.open(pdf_path)
   text_by_page = []
   for page_num, page in enumerate(doc):
       text_by_page.append((page_num, page.get_text()))
   return text_by_page


def preprocess_text(text: str) -> str:
   """Preprocess text by removing extra whitespace, punctuation, and other noise."""
   text = text.lower()  # Convert to lowercase
   text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
   text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
   text = re.sub(r'[^a-zA-Z0-9\s\*\-\•]', '', text)  # Keep bullet characters
   return text.strip()


def extract_tables_from_pdf(pdf_path):
   """Extract tables from a PDF file and return as a list of DataFrames."""
   tables = []
   with pdfplumber.open(pdf_path) as pdf:
       for page in pdf.pages:
           extracted_tables = page.extract_tables()
           for table in extracted_tables:
               # Convert to DataFrame for structured storage
               df = pd.DataFrame(table)
               tables.append(df)
   return tables


def detect_bullet_points(text):
   """Identify and format bullet points and numbered lists in text."""
   bullet_point_patterns = [
       r'^\s*[\*\-•]\s+',   # Match *, -, • followed by space
       r'^\s*\d+\.\s+',     # Match numbered lists like 1., 2., 3.
       r'^\s*[a-zA-Z]\)\s+' # Match lettered lists like a), b), c)
   ]
  
   formatted_lines = []
   for line in text.split("\n"):
       if any(re.match(pattern, line) for pattern in bullet_point_patterns):
           formatted_lines.append(f"- {line.strip()}")  # Convert to a standardized bullet format
       else:
           formatted_lines.append(line.strip())
  
   return "\n".join(formatted_lines)


def extract_captions_from_text(text):
   """Identify possible captions in extracted text."""
   caption_patterns = [
       r'Figure\s\d+[:\.\-]',  # Matches "Figure 1:", "Figure 2.", "Figure 3 -"
       r'Table\s\d+[:\.\-]',   # Matches "Table 1:", "Table 2."
   ]
  
   captions = []
   for line in text.split("\n"):
       if any(re.search(pattern, line, re.IGNORECASE) for pattern in caption_patterns):
           captions.append(line.strip())
  
   return captions


# split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
   """Split text into chunks of approximately chunk_size words with overlap."""
   words = text.split()
   chunks = []
   for i in range(0, len(words), chunk_size - overlap):
       chunk = " ".join(words[i : i + chunk_size])
       chunks.append(chunk)
   return chunks



def process_pdfs(data_dir):
   for file_name in os.listdir(data_dir):
       if file_name.endswith(".pdf"):
           pdf_path = os.path.join(data_dir, file_name)
           text_by_page = extract_text_from_pdf(pdf_path)  # Extract text from PDF
           tables = extract_tables_from_pdf(pdf_path)  # Extract tables separately
          
           for page_num, text in text_by_page:
               cleaned_text = preprocess_text(text)  # Clean text first
               formatted_text = detect_bullet_points(cleaned_text)  # Then format bullet points
               captions = extract_captions_from_text(formatted_text)  # Extract captions
               chunks = split_text_into_chunks(formatted_text)  # Chunk the cleaned, formatted text
               # print(f"  Chunks: {chunks}")
              
               for chunk_index, chunk in enumerate(chunks):
                   # embedding = calculate_embedding(chunk)
                   embedding = get_embedding3(chunk) 
                   store_embedding(
                       file=file_name,
                       page=str(page_num),
                       #chunk=str(chunk_index),
                       chunk=str(chunk),
                       embedding=embedding,
                   )
              
           print(f"-----> Processed {file_name} (Tables: {len(tables)})")



def query_redis(query_text: str):
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance")
        .dialect(2)
    )
    query_text = "Efficient search in vector databases"
    embedding = get_embedding2(query_text)
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    # print(res.docs)

    for doc in res.docs:
        print(f"{doc.id} \n ----> {doc.vector_distance}\n")


def main():
    clear_redis_store()
    create_hnsw_index()

    process_pdfs("/Users/pavithra/Downloads/RagIngestAndSearch-practical-02-pavi_lesrene_zainab/data/ds4300 notes")

    print("\n---Done processing PDFs---\n")
    #query_redis("What is the capital of France?")


if __name__ == "__main__":
    main()
