import os
from pathlib import Path 

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from image_reader import get_image_text
from langchain_core.documents import Document
from typing import List

def load_data() -> List[Document]:
    doc_folder_path = "documents"
    root_path = Path(doc_folder_path)
    documents = []
    for file in os.listdir(root_path):
        doc_path = doc_folder_path+"/"+file
        if file.endswith(".pdf"):
            document=PyPDFLoader(doc_path).load()
        elif file.endswith(".txt"):
            document=TextLoader(doc_path).load()
        elif file.endswith(".docx") or file.endswith(".doc"):
            document=Docx2txtLoader(doc_path).load()
        elif file.endswith(".png") or file.endswith(".jpg"):
            document=get_image_text(doc_path)
        documents.extend(document)
    if len(documents) > 0:
        #print(len(documents))
        return documents
          