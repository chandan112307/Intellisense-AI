# app/main.py

import uuid
import time

# Trigger reload
from fastapi import FastAPI, Request, Security
from fastapi.security import APIKeyHeader
from fastapi.openapi.models import APIKey, APIKeyIn
from fastapi.openapi.utils import get_openapi

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.chat_router import router as chat_router
from app.api.routes.session import router as session_router
from app.api.routes.auth_router import router as auth_router
from app.api.routes.verification_router import router as verification_router
from app.api.routes.ingestion_router import router as ingestion_router
from app.api.routes.evilearn_router import router as evilearn_router
from app.api.routes.admin_router import router as admin_router
from app.api.routes.admin_storage_router import router as admin_storage_router
from app.api.routes.student_knowledge_router import router as student_knowledge_router
from app.api.routes.feedback_router import router as feedback_router

from app.core.logging import log_info, log_error
auth_scheme = APIKeyHeader(name="Authorization", auto_error=False)
app = FastAPI(
    title="IntelliSense AI — Hybrid Agentic RAG Backend",
    version="2.0.0",
    description="Backend powering Query Understanding -> Retrieval -> Response Synthesis + EviLearn Verification",
    swagger_ui_init_oauth={},
    openapi_tags=[
        {"name": "auth", "description": "Authentication"},
        {"name": "session", "description": "Session Manager"},
        {"name": "chat", "description": "Query understanding -> Retrieval -> Response Synthesis"},
        {"name": "ingestion", "description": "File and URL Ingestion"},
        {"name": "evilearn", "description": "EviLearn: Hybrid Verification & Storage-Efficient RAG"},
        {"name": "admin", "description": "Admin Dashboard & System Management"},
        {"name": "student-knowledge", "description": "Student Knowledge Ingestion & Retrieval"},
        {"name": "feedback", "description": "User Feedback Loop System"},
    ]
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(session_router)
app.include_router(auth_router)
app.include_router(verification_router)
app.include_router(ingestion_router)
app.include_router(evilearn_router)
app.include_router(admin_router)
app.include_router(admin_storage_router) # Added include_router
app.include_router(student_knowledge_router)
app.include_router(feedback_router)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Notebook LM - Agentic RAG Backend",
        version="1.0.0",
        description="Backend powering Query Understanding → Retrieval → Response Synthesis",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }

    # Apply JWT globally to protected routes
    for route in openapi_schema["paths"].values():
        for method in route.values():
            method["security"] = [{"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    trace_id = str(uuid.uuid4())
    start_time = time.time()

    # Log incoming request
    log_info(
        f"Incoming request: {request.method} {request.url.path}",
        trace_id=trace_id
    )

    try:
        response = await call_next(request)

    except Exception as e:
        # Log error
        log_error(
            f"Unhandled exception: {str(e)} | path={request.url.path}",
            trace_id=trace_id
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "trace_id": trace_id,
                "details": str(e)
            }
        )

    # Log response time
    elapsed = round((time.time() - start_time) * 1000, 2)

    log_info(
        f"Completed request: {request.method} {request.url.path} ({elapsed} ms)",
        trace_id=trace_id
    )

    # Attach trace ID to response
    response.headers["X-Trace-ID"] = trace_id

    return response


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "message": "Notebook LM Backend Running"
    }

@app.get("/")
async def home():
    return {
        "status": "ok",
        "message": "Intellisense-AI Backend Running"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    trace_id = request.headers.get("X-Trace-ID", "none")

    log_error(
        f"GLOBAL ERROR: {exc} | path={request.url.path}",
        trace_id=trace_id
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "trace_id": trace_id,
            "details": str(exc)
        }
    )


__all__ = ["app"]
