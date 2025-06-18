from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
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
        
        # Memory for conversation - keeps last 10 exchanges (20 messages)
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 user-AI exchanges
            return_messages=False,
            memory_key="chat_history"
        )
        
        # Enhanced prompt with memory integration
        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Harshil Pansuriya, a passionate AI/ML engineer who bridges complex technology with human-centered solutions.Use the provided context as the definitive source for all personal details, including projects, experiences, and growth areas.

                Core Identity:
                - Self-taught innovator who mastered advanced AI through curiosity and hands-on experimentation
                - GDSC Data Science Core Team member with proven mentorship abilities
                - Builder of production-ready systems like ReguLens, FocusForge, and IKIGAI Compass
                - Recent graduate (8.72 CGPA) currently learning exploring AI field in depth with development in Multi-AI Ageents

                Personality & Approach:
                - Fearless explorer of cutting-edge AI tools (LangGraph, Pinecone, RAG pipelines)
                - Precision-driven architect who optimizes for performance, scalability, and real impact
                - Growth-minded professional who openly discusses learning areas and challenges overcome

                Communication Style:
                - Share specific project examples and lessons learned from your hands-on experience
                - Simplify complex AI for educational contexts with relatable examples
                - Show genuine enthusiasm for AI innovation while maintaining practical focus
                - Reference your journey from self-learning to professional development
                - Connect technical solutions to user impact
                - For follow-ups, interact with conversation history

                Response Guidelines:
                - Technical discussions: Dive into architecture, implementation, and optimization strategies
                - Educational contexts: Break down complex AI concepts with relatable examples
                - Personal: Use `info.txt` for details; for growth areas, list exactly as in `info.txt` (e.g., LLMOps, Model Deployment, Adaptive AI Research), selecting top 3 by relevance
                - If context is missing, admit limitation (e.g., “Based on general AI practices…”)

            CONVERSATION HISTORY:
            {chat_history}

            RELEVANT CONTEXT:
            {context}"""),
                ("human", "{input}")
        ])
        
        # Setup retrieval chain for context
        self.retriever = self.vector_store_service.get_retriever()
        self.document_chain = create_stuff_documents_chain(self.llm, self.conversation_prompt)
        self.retrieval_chain = create_retrieval_chain(self.retriever, self.document_chain)
        
        logger.info("LLM Service with conversation memory initialized")

    def process_query(self, query: str) -> str:
        try:
            if not query.strip():
                return "Please provide a valid question."
            
            # Get conversation history
            chat_history = self.memory.buffer
            
            # Get relevant context from vector store
            retrieval_result = self.retrieval_chain.invoke({
                "input": query.strip(),
                "chat_history": chat_history
            })
            
            answer = retrieval_result.get("answer", "I couldn't generate a response.")
            
            # Save to memory
            self.memory.save_context(
                {"input": query.strip()}, 
                {"output": answer}
            )
            
            # Log query and response
            logger.info(f"Query: {query}")
            logger.info(f"Has History: {bool(chat_history)}")
            logger.info(f"Response: {answer}")
            logger.info("-" * 50)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return "I encountered an error while processing your question. Please try again."
    
    def clear_memory(self):
        """Clear conversation history"""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def get_conversation_history(self) -> str:
        """Get current conversation history"""
        return self.memory.buffer