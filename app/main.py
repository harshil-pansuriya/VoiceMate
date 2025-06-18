from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from speech_service import SpeechService
from config.logging import logger
import base64
import uvicorn

app = FastAPI(title="VoiceMate AI with Memory")

speech_service = SpeechService()

@app.get("/")
async def root():
    return {"message": "VoiceMate AI with conversation memory is running"}

@app.post("/process_voice")
async def process_voice(file: UploadFile = File(...)):
    """Process voice with conversation memory"""
    try:
        audio_data = await file.read()
        logger.info(f"Received audio file: {file.filename}")
        
        # Process voice query with memory
        response_text, audio_response = speech_service.process_voice_query(audio_data)
        
        # Encode audio response as base64
        audio_base64 = base64.b64encode(audio_response).decode('utf-8') if audio_response else ""
        
        return JSONResponse(
            content={
                "response_text": response_text,
                "audio_response": audio_base64,
                "has_memory": True
            },
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Error in /process_voice: {str(e)}")
        return JSONResponse(
            content={"error": "Failed to process audio."},
            status_code=500
        )

@app.post("/clear_memory")
async def clear_memory():
    """Clear conversation memory"""
    try:
        speech_service.clear_conversation_memory()
        logger.info("Memory cleared via API")
        return JSONResponse(
            content={"message": "Conversation memory cleared successfully"},
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")
        return JSONResponse(
            content={"error": "Failed to clear memory"},
            status_code=500
        )

@app.get("/memory_status")
async def memory_status():
    """Get current memory status"""
    try:
        history = speech_service.llm_service.get_conversation_history()
        return JSONResponse(
            content={
                "has_conversation": bool(history),
                "conversation_length": len(history) if history else 0
            },
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error getting memory status: {str(e)}")
        return JSONResponse(
            content={"error": "Failed to get memory status"},
            status_code=500
        )

if __name__ == "__main__":
    logger.info("VoiceMate Server with Conversation Memory")
    
    uvicorn.run(
        app,
        host='localhost',
        port=8080,
        log_level='info'
    )