from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from vector_store import VectorStoreService
from config.setting import Config
from config.logging import logger

class LLMService:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=Config.groq_api_key,
            model_name="llama3-70b-8192",
            temperature=0.7,
        )
        
        self.vector_store_service = VectorStoreService()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Harshil Pansuriya, a visionary AI/ML engineer with expertise in Python, Generative AI, LLMs, RAG, and AI agents. Using the provided context about your skills, projects (e.g., FocusForge, ReguLens), and experiences, respond to questions as Harshil would. Reflect his curiosity, precision, and collaborative nature. Provide detailed, technical answers showcasing problem-solving skills and innovative approaches, while keeping responses concise and clear. If the query is vague, ambiguous, or too short (e.g., single words), respond with: "Could you please provide more details or clarify your question? I'm eager to share my insights on AI/ML, RAG, or my projects like ReguLens!" If the context lacks information, admit limitations but offer insights based on your expertise in FastAPI, LangChain, Pinecone, and related tools.

            Context: {context}"""),
            ("human", "{input}")
        ])
        
        retriever = self.vector_store_service.get_retriever()
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        self.chain = create_retrieval_chain(retriever, document_chain)
        
        logger.info("LLM Service initialized")

    def process_query(self, query: str) -> str:
        try:
            if not query.strip():
                return "Please provide a valid question."
            
            response = self.chain.invoke({"input": query.strip()})
            answer = response.get("answer", "I couldn't generate a response.")
            
            # Log full query and response
            logger.info(f"Query: {query}")
            logger.info(f"LLM Response: {answer}")
            logger.info("-" * 50)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return "I encountered an error while processing your question. Please try again."