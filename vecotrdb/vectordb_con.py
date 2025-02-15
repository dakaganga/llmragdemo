import chromadb
import PyPDF2
from sentence_transformers import SentenceTransformer
import service.generative_ai_service
from langchain_community.vectorstores import Chroma
#from vecotrdb.embedder import CustomEmbedder
#from utils.data_load_utils import load_data
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
#custom_embedder=CustomEmbedder()
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

def add_text_to_collection(collection, metadata, doc_id):
    claims=service.generative_ai_service.readClaimFiles()
    claimdocid=1
    for claimtext in claims:
        collection.add(
        documents=[claimtext],
        metadatas=[metadata],
        ids=[claimdocid]
        )
        claimdocid = claimdocid+1
'''        
def add_docs_to_collection(collection):
    document_splitter = RecursiveCharacterTextSplitter(chunk_size=25, chunk_overlap=10)
    document_chunks = document_splitter.split_documents(load_data())
    for count, chunk in enumerate(document_chunks):
        collection.add(
            documents=chunk.page_content,
            metadatas=[{"source": chunk.metadata['source']}],
            ids=[chunk.metadata['source']+ str(count)]
         )
'''
# query collection
def query_collection(collection, query_text):
    print(query_text)
    result = collection.query(
        query_texts=[query_text],
        n_results=2
    )
    return result


def Get_data(Question,DB, embedding):
    persist_directory = DB
    vectorstore =Chroma(persist_directory=persist_directory)
    relevant_documents = vectorstore.similarity_search_with_score(Question)
    sorted_results = sorted(relevant_documents, key=lambda x: x[1])
   # return (sorted_results[0])
    return (relevant_documents)

'''
db4 = Chroma(
    client=client,
    collection_name="my_collection",
    embedding_function=embedding_function,
)
query = "What did the president say about Ketanji Brown Jackson"
docs = db4.similarity_search(query)
print(docs[0].page_content)
'''
# Quering for a RAG application:
# 0. Add embeddings 
# 1. Retrieve Documents
# 2. Rank (if not done by retrival function)
# 3. Extract and aggregate the information 
# 4. Prepare context and format for LLM 
# 5. Use llm to generate a responce 
# 6. Handle output (display or store)

def getCollection():
    collection_name = "my_collection"
    #client.delete_collection(name=collection_name)
    collection = client.get_or_create_collection(name=collection_name)
    return collection

def main():
    # Assuming you have a client setup as shown in previous examples
    collection_name = "my_collection"
    collection = client.get_or_create_collection(name=collection_name)

    # Add a PDF file to the collection
    pdf_path = "./dummy.pdf"  # Update this path to your PDF file
    metadata = {"type": "text"}
    doc_id = "sample_doc"
    #add_pdf_to_collection(collection, pdf_path, metadata, doc_id)
    add_text_to_collection(collection,metadata)
    # Query the collection
    query_text = "list oout the patient names who had comonoscopy?"
    result = query_collection(collection, query_text)

    # Print the results
    print("Query Results:")
    print(result)

if __name__ == "__main__":
    main()