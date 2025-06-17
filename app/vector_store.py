import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.setting import Config
from config.logging import logger

from typing import List

class VectorStoreService:
    def __init__(self):
        self.pc= Pinecone(api_key= Config.pinecone_api_key)
        self.index= self.pc.Index(Config.pinecone_index)
        
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        logger.info("VectorStoreService initialized")
        
    def load_and_chunk_data(self) -> List[str]:
        try: 
            with open('data/info.txt', 'r') as file:
                text= file.read()
                
        except Exception as e:
            print(f"File not found: {str(e)}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Loaded and chunked {len(chunks)} text chunks")
        return chunks
    
    def initialize_vector_store(self):
        chunks= self.load_and_chunk_data()
        embeddings= self.embeddings.embed_documents(chunks)
        
        vectors= [
            {
                "id": f"chunk_{i}",
                "values":embedding,
                'metadata': {'text' :chunk},
            }
            for i ,(chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        self.index.upsert(vectors= vectors, namespace="Person-info")
        logger.info(f"Stored {len(vectors)} embeddings in Pinecone namespace 'Person-info'")
        
    def get_retriever(self, query:str) -> List[str]:
        vector_store = Pinecone(
            index=self.index,
            embedding=self.embeddings,
            namespace="candidate_info",
            text_key="text"
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        logger.info("Retriever initialized")
        return retriever