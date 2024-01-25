import chromadb
import PyPDF2
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    return model.encode(text)

# Local db mode
client = chromadb.Client()

# Client-server mode. Run using chroma run --path /db_path
# chroma_client = chromadb.HttpClient(host='localhost', port=8000)

collection = client.create_collection(name="my_collection")
# or
# collection = client.get_or_create_collection(name="my_collection")


# add document to collections
def add_pdf_to_collection(collection, pdf_path, metadata, doc_id):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''.join([page.extract_text() for page in reader.pages])

    collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[doc_id]
    )

# query collection
def query_collection(collection, query_text):
    result = collection.query(
        query_texts=[query_text],
        n_results=1
    )
    return result

# Quering for a RAG application:
# 0. Add embeddings 
# 1. Retrieve Documents
# 2. Rank (if not done by retrival function)
# 3. Extract and aggregate the information 
# 4. Prepare context and format for LLM 
# 5. Use llm to generate a responce 
# 6. Handle output (display or store)



def main():
    # Assuming you have a client setup as shown in previous examples
    collection_name = "my_collection"
    collection = client.get_or_create_collection(name=collection_name)

    # Add a PDF file to the collection
    pdf_path = "./dummy.pdf"  # Update this path to your PDF file
    metadata = {"type": "document"}
    doc_id = "sample_doc"
    add_pdf_to_collection(collection, pdf_path, metadata, doc_id)

    # Query the collection
    query_text = "What does the document say?"
    result = query_collection(collection, query_text)

    # Print the results
    print("Query Results:")
    print(result)

if __name__ == "__main__":
    main()