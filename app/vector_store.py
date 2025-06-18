from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.base import Embeddings
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from config.setting import Config
from config.logging import logger
from typing import List

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

class VectorStoreService:
    def __init__(self):
        self.pc = Pinecone(api_key=Config.pinecone_api_key)
        self.index = self.pc.Index(Config.pinecone_index)
        self.embeddings = SentenceTransformerEmbeddings()
        self._vector_store = None
        logger.info("Vector store service initialized")

    def load_and_chunk_data(self) -> List[str]:
        with open("data/info.txt", "r") as f:
            text = f.read()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,    # Larger chunks for better context
            chunk_overlap=50, # More overlap for continuity
            length_function=len
        )
        chunks = splitter.split_text(text)
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks

    def initialize_vector_store(self):
        chunks = self.load_and_chunk_data()
        embeddings_list = self.embeddings.embed_documents(chunks)
        
        vectors = [
            {
                'id': f"chunk_{i}",
                'values': embedding,
                'metadata': {'text': chunk}
            }
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_list))
        ]
        
        self.index.upsert(vectors=vectors, namespace="candidate_info")
        logger.info(f"Stored {len(vectors)} vectors in Pinecone")

    def get_retriever(self):
        if self._vector_store is None:
            self._vector_store = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                namespace="candidate_info",
                text_key="text"
            )
        
        return self._vector_store.as_retriever(
            search_kwargs={"k": 3}  # Retrieve more context for better answers
        )