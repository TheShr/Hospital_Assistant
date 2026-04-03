"""
RAG-based Hospital Patient Query Assistant
Production-grade FastAPI backend
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import logging

from api.routes import router
from core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("🏥 Hospital RAG Assistant starting up...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Hospital RAG Assistant API",
    description="AI-powered Patient Query Assistant using RAG pipeline",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Routes
app.include_router(router, prefix="/api/v1")
from core.config import settings

print("KEY:", settings.SUPABASE_SERVICE_KEY)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Hospital RAG Assistant"}
