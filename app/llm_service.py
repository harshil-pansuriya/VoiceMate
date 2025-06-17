from langchain_groq import Groq
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory

from vector_store import VectorStoreService
from config.setting import Config
from config.logging import logger

class LLMService:
    def __init__(self):
        self.llm= Groq(
            api_key=Config.groq_api_key,
            model_name="llama3-8b-8192",
            temperature=0.7,
        )
        
        self.vector_store_service = VectorStoreService()
        
        self.memory= ConversationBufferMemory(
            memory_key='chat_memory',
            retrun_messages=True
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a bot representing me, reflecting my personality, skills, and thought process. 
            Use the following context and conversation history to answer the question as I would, 
            showcasing my problem-solving skills, thinking ability, and personal traits. Provide 
            thoughtful, detailed responses that demonstrate how I approach challenges and reflect my 
            unique perspective.
            """),
            ("human", "Conversation History:\n{chat_history}\n\nQuestion:\n{input}"),
        ])
        
        retriever = self.vector_store_service.get_retriever()
        qna_chain = create_stuff_documents_chain(self.llm, prompt)
        self.chain = create_retrieval_chain(retriever, qna_chain)
        logger.info("LLMRAGService initialized")
        
    def process_query(self, query:str) -> str:
        response= self.chain.invoke({
            "input": query,
            "chat_history": self.memory.load_memory_variables({})['chat_history']
        })
        
        self.memory.save_context({"input": query}, {"output": response["answer"]})
        logger.debug(f"Processed query: {query}, Response: {response['answer']}")
        return response["answer"]