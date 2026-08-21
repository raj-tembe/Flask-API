"""
AI‑Powered Code Review & Refactoring API

A production‑grade service that accepts code submissions, runs an asynchronous
analysis (static + LLM) to suggest improvements, detect bugs, and even propose
refactoring. All data is persisted in a SQL database (SQLite/PostgreSQL) with full CRUD.

Features:
  - Submit code snippets (with language, file path, optional context)
  - Background review worker (mock LLM – replace with your own)
  - Store review results: severity, suggestions, line numbers, etc.
  - Manual comments can be added by team members
  - Full CRUD: list, get, update, delete reviews
  - Pagination and filtering (by language, status, team)
  - Team‑based API key authentication
  - Admin endpoints for system stats
  - SQLAlchemy ORM (async ready – uses async SQLAlchemy for non‑blocking IO)
"""

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, status, Query, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text,
    ForeignKey, Index, Enum as SQLEnum, func
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, selectinload
from sqlalchemy.future import select
from sqlalchemy.sql import and_, or_, desc

load_dotenv()

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Security
    admin_api_key: str = Field(..., description="Admin API key")
    algorithm: str = Field("HS256")

    # Database
    database_url: str = Field("sqlite+aiosqlite:///./code_reviews.db")

    # Rate limiting
    rate_limit_per_minute: int = Field(60)

    # Logging
    log_level: str = Field("INFO")

    # LLM (mock) settings – replace with real API keys
    llm_api_key: Optional[str] = None
    llm_model: str = Field("gpt-4")

    # Worker
    worker_sleep_seconds: float = Field(1.0)

    try:
        settings = Settings()
    except Exception as e:
        raise RuntimeError(f"Missing ADMIN_API_KEY in .env. Details: {e}")

settings = Settings()

# Setup logging
logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
logger = logging.getLogger("code_review_api")

# --------------------------------------------------------------------------
# Database Setup (Async SQLAlchemy)
# --------------------------------------------------------------------------

engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)
Base = declarative_base()

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class ReviewStatus(str, Enum):
    PENDING = "pending"         # submitted, not yet processed
    PROCESSING = "processing"   # analysis running
    COMPLETED = "completed"     # analysis done
    FAILED = "failed"           # analysis error
    REVIEWED = "reviewed"       # human has reviewed and possibly added comments

class Review(Base):
    __tablename__ = "reviews"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    team = Column(String(100), nullable=False, index=True)
    code = Column(Text, nullable=False)
    language = Column(String(50), nullable=False, index=True)
    file_path = Column(String(255), nullable=True)
    context = Column(Text, nullable=True)

    status = Column(SQLEnum(ReviewStatus), default=ReviewStatus.PENDING, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Analysis results
    score = Column(Float, nullable=True)                 # 0-100
    summary = Column(Text, nullable=True)
    suggestions = Column(Text, nullable=True)            # JSON array of suggestions

    # Optional refactoring suggestions (code diff)
    refactored_code = Column(Text, nullable=True)

    processing_time_ms = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    comments = relationship("Comment", back_populates="review", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_team_status", "team", "status"),
        Index("idx_language_created", "language", "created_at"),
    )

class Comment(Base):
    __tablename__ = "comments"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    review_id = Column(String(36), ForeignKey("reviews.id"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    line_start = Column(Integer, nullable=True)
    line_end = Column(Integer, nullable=True)
    created_by = Column(String(100), nullable=False)    # team or user
    created_at = Column(DateTime, default=datetime.utcnow)

    review = relationship("Review", back_populates="comments")

# --------------------------------------------------------------------------
# Pydantic Schemas
# --------------------------------------------------------------------------

class CodeSubmitRequest(BaseModel):
    code: str = Field(..., min_length=1)
    language: str = Field(..., min_length=1, max_length=50)
    file_path: Optional[str] = Field(None, max_length=255)
    context: Optional[str] = None

    @validator('language')
    def validate_language(cls, v):
        allowed = ['python', 'javascript', 'typescript', 'java', 'go', 'rust', 'cpp', 'csharp', 'ruby', 'php']
        if v.lower() not in allowed:
            raise ValueError(f"Language '{v}' not supported. Allowed: {allowed}")
        return v.lower()

class ReviewResponse(BaseModel):
    id: str
    team: str
    language: str
    file_path: Optional[str]
    context: Optional[str]
    status: ReviewStatus
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
    score: Optional[float]
    summary: Optional[str]
    suggestions: Optional[Dict[str, Any]]   # JSON parsed
    refactored_code: Optional[str]
    error_message: Optional[str]
    processing_time_ms: Optional[float]
    comments: List[Dict[str, Any]] = []

    class Config:
        from_attributes = True

class ReviewListResponse(BaseModel):
    items: List[ReviewResponse]
    total: int
    limit: int
    offset: int

class CommentCreateRequest(BaseModel):
    content: str = Field(..., min_length=1)
    line_start: Optional[int] = Field(None, ge=0)
    line_end: Optional[int] = Field(None, ge=0)

    @validator('line_end')
    def validate_line_range(cls, v, values):
        if 'line_start' in values and values['line_start'] is not None and v is not None:
            if v < values['line_start']:
                raise ValueError('line_end must be >= line_start')
        return v

class AdminStatsResponse(BaseModel):
    total_reviews: int
    by_status: Dict[str, int]
    by_language: Dict[str, int]
    avg_score: Optional[float]
    avg_processing_time_ms: Optional[float]
    total_comments: int

# --------------------------------------------------------------------------
# Authentication & Rate Limiting
# --------------------------------------------------------------------------

API_KEYS = {
    "test_team_123": "engineering",
    "test_team_456": "data_science",
    "test_team_789": "backend",
    settings.admin_api_key: "admin"
}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)

# Simple in‑memory rate limiter (replace with Redis in production)
rate_limit_store: Dict[str, List[float]] = {}
rate_lock = asyncio.Lock()

async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    team = API_KEYS.get(api_key)
    if not team:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return team

async def verify_admin_key(admin_key: str = Security(admin_key_header)) -> str:
    if not admin_key:
        raise HTTPException(status_code=401, detail="Missing X-Admin-Key header")
    if admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return "admin"

async def rate_limit_check(team: str) -> bool:
    """Simple per-minute rate limit using async lock."""
    async with rate_lock:
        now = time.time()
        window = now - 60
        if team not in rate_limit_store:
            rate_limit_store[team] = []
        # Clean old entries
        rate_limit_store[team] = [t for t in rate_limit_store[team] if t > window]
        if len(rate_limit_store[team]) >= settings.rate_limit_per_minute:
            return False
        rate_limit_store[team].append(now)
        return True

# --------------------------------------------------------------------------
# Background Worker (mock LLM analysis)
# --------------------------------------------------------------------------

async def analyze_code(review_id: str, code: str, language: str, context: str) -> Dict[str, Any]:
    """
    This is the core analysis function. In production, replace with a call to an LLM
    (OpenAI, Anthropic, etc.) with a suitable prompt for code review.
    For now, we use a simple rule‑based mock to demonstrate the pipeline.
    """
    # Simulate processing time
    await asyncio.sleep(2)

    # Mock analysis based on simple heuristics
    suggestions = []
    score = 85  # default

    # Example: check for missing docstring
    if '"""' not in code and "'''" not in code:
        suggestions.append({
            "severity": "info",
            "line": 1,
            "message": "Add docstring to explain the purpose of the code.",
            "suggestion": "Consider adding a docstring at the top."
        })
        score -= 5

    # Check for very long lines
    lines = code.split('\n')
    for i, line in enumerate(lines, 1):
        if len(line) > 100:
            suggestions.append({
                "severity": "warning",
                "line": i,
                "message": f"Line {i} is too long ({len(line)} chars).",
                "suggestion": "Break the line to improve readability."
            })
            score -= 2

    # Language-specific checks
    if language == "python":
        if "except:" in code and "except Exception" not in code:
            suggestions.append({
                "severity": "error",
                "line": None,
                "message": "Bare except clause found.",
                "suggestion": "Use 'except Exception as e:' to catch specific exceptions."
            })
            score -= 10
    elif language == "javascript":
        if "var " in code:
            suggestions.append({
                "severity": "warning",
                "line": None,
                "message": "Using 'var' is outdated.",
                "suggestion": "Use 'let' or 'const' instead."
            })
            score -= 3

    # If no suggestions, add a positive one
    if not suggestions:
        suggestions.append({
            "severity": "info",
            "line": None,
            "message": "No major issues detected.",
            "suggestion": "Keep up the good practice!"
        })

    # Ensure score between 0 and 100
    score = max(0, min(100, score))

    # Generate a mock refactored version (just add a comment)
    refactored_code = code + "\n# TODO: Consider improving error handling"

    summary = f"Analysis complete. Score: {score}/100. Found {len(suggestions)} suggestions."

    return {
        "score": score,
        "summary": summary,
        "suggestions": suggestions,
        "refactored_code": refactored_code
    }

async def process_review(review_id: str, db: AsyncSession):
    """Fetch review, run analysis, update results."""
    # Get review
    stmt = select(Review).where(Review.id == review_id)
    result = await db.execute(stmt)
    review = result.scalar_one_or_none()
    if not review:
        return

    try:
        review.status = ReviewStatus.PROCESSING
        await db.commit()

        start = time.perf_counter()
        analysis = await analyze_code(
            review_id=review.id,
            code=review.code,
            language=review.language,
            context=review.context or ""
        )
        processing_time = (time.perf_counter() - start) * 1000

        # Update review with results
        review.status = ReviewStatus.COMPLETED
        review.score = analysis["score"]
        review.summary = analysis["summary"]
        review.suggestions = analysis["suggestions"]  # JSON serializable
        review.refactored_code = analysis["refactored_code"]
        review.processing_time_ms = processing_time
        review.completed_at = datetime.utcnow()
        await db.commit()
    except Exception as e:
        logger.exception(f"Review {review_id} processing failed")
        review.status = ReviewStatus.FAILED
        review.error_message = str(e)
        await db.commit()

# --------------------------------------------------------------------------
# Worker loop
# --------------------------------------------------------------------------

async def background_worker():
    """Periodically poll for pending reviews and process them."""
    logger.info("Background worker started")
    while True:
        try:
            async with AsyncSessionLocal() as db:
                # Find pending reviews
                stmt = select(Review).where(Review.status == ReviewStatus.PENDING).order_by(Review.created_at)
                result = await db.execute(stmt)
                pending = result.scalars().all()

                if not pending:
                    await asyncio.sleep(settings.worker_sleep_seconds)
                    continue

                # Process one review at a time to avoid concurrency issues
                for review in pending[:1]:  # process one per loop iteration
                    # We'll run analysis in a separate task to not block the loop
                    asyncio.create_task(process_review(review.id, db))
                # Wait a bit before checking again
                await asyncio.sleep(1)
        except Exception as e:
            logger.exception("Worker error")
            await asyncio.sleep(5)

# --------------------------------------------------------------------------
# FastAPI App
# --------------------------------------------------------------------------

app = FastAPI(
    title="AI Code Review API",
    description="Submit code for AI‑powered review, store results, and manage comments with full CRUD.",
    version="1.0.0"
)

# Startup: create tables and start worker
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Start background worker as a background task
    asyncio.create_task(background_worker())

@app.on_event("shutdown")
async def shutdown():
    await engine.dispose()

# --------------------------------------------------------------------------
# Dependency: get DB session
# --------------------------------------------------------------------------

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

# --------------------------------------------------------------------------
# CRUD Endpoints
# --------------------------------------------------------------------------

@app.post("/api/v1/code", response_model=ReviewResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_code(
    req: CodeSubmitRequest,
    team: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db)
):
    """Submit code for review. The review will be processed asynchronously."""
    # Rate limit
    if not await rate_limit_check(team):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Create review
    review = Review(
        team=team,
        code=req.code,
        language=req.language,
        file_path=req.file_path,
        context=req.context,
        status=ReviewStatus.PENDING
    )
    db.add(review)
    await db.commit()
    await db.refresh(review)

    # Return the review (it will be picked up by the worker)
    response = ReviewResponse.from_orm(review)
    return response

@app.get("/api/v1/code", response_model=ReviewListResponse)
async def list_reviews(
    team: str = Depends(verify_api_key),
    language: Optional[str] = Query(None, description="Filter by language"),
    status: Optional[ReviewStatus] = Query(None, description="Filter by status"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """List reviews for the current team with optional filters and pagination."""
    filters = [Review.team == team]
    if language:
        filters.append(Review.language == language)
    if status:
        filters.append(Review.status == status)

    # Count total
    count_stmt = select(func.count()).select_from(Review).where(and_(*filters))
    total = await db.scalar(count_stmt)

    # Fetch items
    stmt = select(Review).where(and_(*filters)).order_by(desc(Review.created_at)).offset(offset).limit(limit)
    result = await db.execute(stmt)
    reviews = result.scalars().all()

    # Load comments for each review (eagerly)
    items = []
    for rev in reviews:
        # load comments
        await db.refresh(rev, attribute_names=["comments"])
        items.append(ReviewResponse.from_orm(rev))

    return ReviewListResponse(items=items, total=total, limit=limit, offset=offset)

@app.get("/api/v1/code/{review_id}", response_model=ReviewResponse)
async def get_review(
    review_id: str,
    team: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific review by ID with its comments."""
    stmt = select(Review).where(Review.id == review_id).options(selectinload(Review.comments))
    result = await db.execute(stmt)
    review = result.scalar_one_or_none()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    if review.team != team:
        raise HTTPException(status_code=403, detail="Access denied")
    return ReviewResponse.from_orm(review)

@app.put("/api/v1/code/{review_id}", response_model=ReviewResponse)
async def update_review(
    review_id: str,
    status: Optional[ReviewStatus] = None,
    team: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db)
):
    """
    Update review status (e.g., mark as reviewed). Only status can be updated via this endpoint.
    For a full update, consider a more general endpoint.
    """
    stmt = select(Review).where(Review.id == review_id)
    result = await db.execute(stmt)
    review = result.scalar_one_or_none()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    if review.team != team:
        raise HTTPException(status_code=403, detail="Access denied")

    if status is not None:
        review.status = status
        review.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(review)

    # Reload with comments
    await db.refresh(review, attribute_names=["comments"])
    return ReviewResponse.from_orm(review)

@app.delete("/api/v1/code/{review_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_review(
    review_id: str,
    team: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db)
):
    """Delete a review and all its comments."""
    stmt = select(Review).where(Review.id == review_id)
    result = await db.execute(stmt)
    review = result.scalar_one_or_none()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    if review.team != team:
        raise HTTPException(status_code=403, detail="Access denied")

    await db.delete(review)
    await db.commit()

@app.post("/api/v1/code/{review_id}/comments", response_model=Dict[str, Any])
async def add_comment(
    review_id: str,
    req: CommentCreateRequest,
    team: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db)
):
    """Add a manual comment to a review."""
    # Check if review exists and belongs to team
    stmt = select(Review).where(Review.id == review_id)
    result = await db.execute(stmt)
    review = result.scalar_one_or_none()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    if review.team != team:
        raise HTTPException(status_code=403, detail="Access denied")

    comment = Comment(
        review_id=review_id,
        content=req.content,
        line_start=req.line_start,
        line_end=req.line_end,
        created_by=team
    )
    db.add(comment)
    await db.commit()
    await db.refresh(comment)

    return {"message": "Comment added", "comment_id": comment.id}

@app.delete("/api/v1/code/{review_id}/comments/{comment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_comment(
    review_id: str,
    comment_id: str,
    team: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db)
):
    """Delete a comment. Only the team that owns the review can delete comments."""
    # Verify review ownership
    stmt = select(Review).where(Review.id == review_id)
    result = await db.execute(stmt)
    review = result.scalar_one_or_none()
    if not review or review.team != team:
        raise HTTPException(status_code=403, detail="Access denied")

    stmt = select(Comment).where(Comment.id == comment_id, Comment.review_id == review_id)
    result = await db.execute(stmt)
    comment = result.scalar_one_or_none()
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")

    await db.delete(comment)
    await db.commit()

# --------------------------------------------------------------------------
# Admin Endpoints
# --------------------------------------------------------------------------

@app.get("/api/v1/admin/stats", response_model=AdminStatsResponse)
async def admin_stats(
    admin: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db)
):
    """Get system-wide statistics. Admin only."""
    # Total reviews
    total_reviews = await db.scalar(select(func.count()).select_from(Review))

    # By status
    status_stmt = select(Review.status, func.count()).group_by(Review.status)
    status_result = await db.execute(status_stmt)
    by_status = {status: count for status, count in status_result.all()}

    # By language
    lang_stmt = select(Review.language, func.count()).group_by(Review.language)
    lang_result = await db.execute(lang_stmt)
    by_language = {lang: count for lang, count in lang_result.all()}

    # Average score (only for completed reviews)
    avg_score = await db.scalar(select(func.avg(Review.score)).where(Review.status == ReviewStatus.COMPLETED))

    # Average processing time
    avg_time = await db.scalar(select(func.avg(Review.processing_time_ms)).where(Review.processing_time_ms.isnot(None)))

    # Total comments
    total_comments = await db.scalar(select(func.count()).select_from(Comment))

    return AdminStatsResponse(
        total_reviews=total_reviews or 0,
        by_status=by_status,
        by_language=by_language,
        avg_score=float(avg_score) if avg_score else None,
        avg_processing_time_ms=float(avg_time) if avg_time else None,
        total_comments=total_comments or 0
    )

# --------------------------------------------------------------------------
# Error Handlers
# --------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {"error": {"status_code": exc.status_code, "detail": exc.detail, "path": request.url.path}}

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.exception("Unhandled exception")
    return {"error": {"status_code": 500, "detail": "Internal server error", "path": request.url.path}}

# --------------------------------------------------------------------------
# Main Entry
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting AI Code Review API with database: {settings.database_url}")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=settings.log_level.lower())
