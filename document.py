from langchain.document_loaders import TextLoader, DirectoryLoader, UnstructuredPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from config import Config
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.docstore.document import Document
from typing import Any, Dict, List

def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            separators=["\n\n", "\n","\n\t"],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks

class DocumentService(object):
    def __init__(self):

        self.config = Config.vector_store_path
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.embedding_model_name)
        self.docs_path = Config.docs_path
        self.vector_store_path = Config.vector_store_path
        self.vector_store = None

    def init_source_vector(self):
        """
        Initialize local knowledge base vector
        :return:
        """
        with open('resource/txt/res.txt','r') as file:
            text=file.read()

        text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=500,
            chunk_overlap=0,
            length_function=len,

        )

        chunks = text_splitter.split_text(text)


        self.vector_store = FAISS.from_texts(chunks, self.embeddings)
    
        self.vector_store.save_local(self.vector_store_path)

    def load_vector_store(self):
        self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings)



if __name__ == '__main__':
    s = DocumentService()
    s.init_source_vector()