from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from speech_service import SpeechService
from config.logging import logger
import base64
import uvicorn

app = FastAPI(title="Voice Bot API")

speech_service = SpeechService()

@app.get("/")
async def root():
    # Root endpoint for server status
    return {"message": "VoiceMate is running"} 

@app.post("/process_voice")
async def process_voice(file: UploadFile = File(...)):
    try:
        # Read audio file
        audio_data = await file.read()
        logger.info(f"Received audio file: {file.filename}")
        
        # Process voice query
        response_text, audio_response = speech_service.process_voice_query(audio_data)
        
        # Encode audio response as base64
        audio_base64 = base64.b64encode(audio_response).decode('utf-8') if audio_response else ""
        
        # Return JSON response
        return JSONResponse(
            content={
                "response_text": response_text,
                "audio_response": audio_base64
            },
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Error in /process_voice: {str(e)}")
        return JSONResponse(
            content={"error": "Failed to process audio."},
            status_code=500
        )
    
    
if __name__ == "__main__":
    logger.info("VoiceMate Server")
    
    uvicorn.run(
        app,
        host='localhost',
        port=8080,
        log_level='info'
    )