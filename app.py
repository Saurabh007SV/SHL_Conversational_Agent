import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modular components
from models import ChatRequest, ChatResponse
from search import SemanticSearch
from llm import LLMAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Initialize core services
logger.info("Starting SemanticSearch initialization...")
search_service = SemanticSearch()

logger.info("Starting LLMAgent initialization...")
llm_agent = LLMAgent()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        # 1. Extract the full conversational context
        user_query = " ".join([m.content for m in request.messages if m.role == 'user'])
        
        # 2. Retrieve relevant items from catalog
        retrieved_catalog = search_service.search(query=user_query, top_k=30)
        
        # 3. Generate response using LLM
        response = llm_agent.generate_chat_response(messages=request.messages, retrieved_catalog=retrieved_catalog)
        
        return response
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
