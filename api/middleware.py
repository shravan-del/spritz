from fastapi import Request
from fastapi.responses import JSONResponse
import logging
from typing import Callable
import traceback

logger = logging.getLogger(__name__)

async def error_handler_middleware(request: Request, call_next: Callable):
    try:
        return await call_next(request)
    except Exception as e:
        # Log the full error with traceback
        logger.error(f"Error processing request: {str(e)}\n{traceback.format_exc()}")
        
        # Return a user-friendly error response
        return JSONResponse(
            status_code=500,
            content={
                "error": "An error occurred while processing your request",
                "detail": str(e),
                "type": "internal_server_error"
            }
        ) 