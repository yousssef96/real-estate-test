# ===================================================================
# Libraries
# ===================================================================

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app import ask_real_estate_bot, DatabaseManager


# Initialize FastAPI application
app = FastAPI(title="Real estate AI Assistant API",
              description="AI API for assiting customers with finding their property requests"
              )

# === HEALTH CHECK ENDPOINT ===
@app.get("/")
def root() -> dict:
    """
    
    Health check endpoint for monitoring and load balancer health check.
    
    Returns:
        dict: A simple status message.
    """
    return {"status": "ok"}

# === REQUEST DATA SCHEMA ===
# Pydantic model for automatic validation and API documentation
class ChatRequest(BaseModel):
    query: str


# === MAIN PREDICTION API ENDPOINT ===
@app.post("/chat")
async def chat_endpoint(request : ChatRequest) -> StreamingResponse:
    """
    Exposes the Real Estate Bot as a streaming API endpoint.

    Args:
        request : An object containing the user's string query.

    Returns:
        StreamingResponse: A stream of text chunks from the AI.
    """
    try:
        def stream_chunks():
            # Iterates through the generated response
            for chunk in ask_real_estate_bot(request.query):
                yield chunk

        return StreamingResponse(stream_chunks(), media_type="text/plain")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat Error: {str(e)}")
    

@app.post("/sync")
async def sync_database() -> dict:
    """
    Triggers the synchronization between Airtable and MongoDB.

    Returns:
        dict: Success message or failure details.
    """
    try:
        success = DatabaseManager.sync_airtable_to_mongodb()
        if success:
            return {"status": "success", "message": "Database synced successfully"}
        return {"status": "warning", "message": "Sync completed but no records were processed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync Error: {str(e)}")