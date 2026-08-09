"""
FastAPI structured data extraction: turns unstructured text into validated
JSON via an LLM's structured-output mode.

Two ways to use it:
  1. Built-in schemas (`/api/extract/{schema_name}`) -- contact info,
     invoices/receipts, support tickets, and calendar events. These are
     bound once at startup (langchain's with_structured_output over a
     fixed Pydantic model), so repeated calls don't pay setup cost.
  2. A custom JSON Schema you supply per-request
     (`/api/extract/custom`) -- for one-off extraction needs without
     writing a Pydantic model first. Necessarily bound per-request since
     the schema varies every call.

Extraction is inherently unreliable for text that doesn't actually contain
a field's value, so every result reports `completeness`: the fraction of
the schema's fields the model actually populated (vs. left null) --
that's the signal for "does this result need a human to check it", not
just whether the API call itself succeeded.

Batch requests process each text independently and don't let one bad item
fail the rest of the batch -- a per-item error is recorded instead.

Setup:
    pip install fastapi "uvicorn[standard]" pydantic-settings \
        langchain-google-genai python-dotenv

    .env:
        GOOGLE_API_KEY=your_gemini_api_key_here

Run:
    python structured_extraction_api.py
    # or: uvicorn structured_extraction_api:app --reload --port 8000

Interactive docs: http://localhost:8000/docs

Example:
    curl -X POST http://localhost:8000/api/extract/contact \
        -H "Content-Type: application/json" \
        -d '{"texts": ["Hi, Im Jane Doe, VP of Engineering at Acme Corp. Reach me at jane@acme.com."]}'

    curl -X POST http://localhost:8000/api/extract/custom \
        -H "Content-Type: application/json" \
        -d '{
          "text": "The blue whale can weigh up to 200 tons and grow to 100 feet long.",
          "json_schema": {
            "title": "AnimalFacts",
            "type": "object",
            "properties": {
              "animal": {"type": "string"},
              "max_weight_tons": {"type": "number"},
              "max_length_feet": {"type": "number"}
            },
            "required": ["animal"]
          }
        }'
"""

import os
from contextlib import asynccontextmanager
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str = Field(..., description="Gemini API key")
    llm_model: str = Field("gemini-2.5-flash")
    max_texts_per_request: int = Field(50)


try:
    settings = Settings()
except Exception as e:
    raise RuntimeError(f"Missing required configuration. Set GOOGLE_API_KEY (in .env or the environment). Details: {e}")

os.environ["GOOGLE_API_KEY"] = settings.google_api_key


# --------------------------------------------------------------------------
# Built-in extraction schemas
# --------------------------------------------------------------------------

class ContactInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None


class InvoiceData(BaseModel):
    invoice_number: Optional[str] = None
    vendor_name: Optional[str] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = None
    due_date: Optional[str] = None
    line_items: List[str] = Field(default_factory=list)


class SupportTicket(BaseModel):
    summary: str
    category: Literal["billing", "technical", "account", "feature_request", "other"]
    priority: Literal["low", "medium", "high", "urgent"]
    sentiment: Literal["positive", "neutral", "negative", "frustrated"]
    action_items: List[str] = Field(default_factory=list)


class EventDetails(BaseModel):
    title: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    location: Optional[str] = None
    attendees: List[str] = Field(default_factory=list)


SCHEMA_REGISTRY: Dict[str, type] = {
    "contact": ContactInfo,
    "invoice": InvoiceData,
    "support_ticket": SupportTicket,
    "event": EventDetails,
}

EXTRACTION_INSTRUCTION = (
    "Extract the requested fields from the text below. Leave a field null "
    "if the text doesn't contain that information -- do not guess or "
    "invent values.\n\nText:\n{text}"
)


def completeness_of(data: dict, field_names: List[str]) -> float:
    if not field_names:
        return 1.0
    present = sum(1 for f in field_names if data.get(f) is not None)
    return present / len(field_names)


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------

resources: Dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = ChatGoogleGenerativeAI(model=settings.llm_model, temperature=0.0)
    resources["llm"] = llm
    resources["extractors"] = {name: llm.with_structured_output(schema) for name, schema in SCHEMA_REGISTRY.items()}
    yield
    resources.clear()


app = FastAPI(
    title="Structured Data Extraction API",
    description="Extracts validated structured JSON from unstructured text via LLM structured output, using built-in schemas or a custom one you supply per request.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="One or more texts to extract from")


class ExtractionResult(BaseModel):
    text: str
    data: Optional[dict]
    completeness: float
    error: Optional[str]


class BatchExtractResponse(BaseModel):
    schema_name: str
    results: List[ExtractionResult]


class SchemaInfo(BaseModel):
    name: str
    json_schema: dict


class SchemaListResponse(BaseModel):
    schemas: List[SchemaInfo]


class CustomExtractRequest(BaseModel):
    text: str = Field(..., min_length=1)
    json_schema: dict = Field(..., description="A JSON Schema object describing the fields to extract")


class CustomExtractResponse(BaseModel):
    data: Optional[dict]
    completeness: float
    error: Optional[str]


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/schemas", response_model=SchemaListResponse)
def list_schemas():
    return SchemaListResponse(schemas=[
        SchemaInfo(name=name, json_schema=schema.model_json_schema())
        for name, schema in SCHEMA_REGISTRY.items()
    ])


@app.post("/api/extract/custom", response_model=CustomExtractResponse)
def extract_custom(payload: CustomExtractRequest):
    llm = resources["llm"]
    field_names = list(payload.json_schema.get("properties", {}).keys())

    try:
        extractor = llm.with_structured_output(payload.json_schema)
        result = extractor.invoke(EXTRACTION_INSTRUCTION.format(text=payload.text))
        data = result if isinstance(result, dict) else dict(result)
        return CustomExtractResponse(data=data, completeness=completeness_of(data, field_names), error=None)
    except Exception as e:
        return CustomExtractResponse(data=None, completeness=0.0, error=str(e))


@app.post("/api/extract/{schema_name}", response_model=BatchExtractResponse)
def extract(schema_name: str, payload: ExtractRequest):
    if schema_name not in SCHEMA_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown schema '{schema_name}'. Available: {', '.join(SCHEMA_REGISTRY.keys())}",
        )

    if len(payload.texts) > settings.max_texts_per_request:
        raise HTTPException(
            status_code=400,
            detail=f"Too many texts ({len(payload.texts)}); max is {settings.max_texts_per_request} per request.",
        )

    extractor = resources["extractors"][schema_name]
    field_names = list(SCHEMA_REGISTRY[schema_name].model_fields.keys())

    results = []
    for text in payload.texts:
        try:
            extracted = extractor.invoke(EXTRACTION_INSTRUCTION.format(text=text))
            data = extracted.model_dump()
            results.append(ExtractionResult(text=text, data=data, completeness=completeness_of(data, field_names), error=None))
        except Exception as e:
            results.append(ExtractionResult(text=text, data=None, completeness=0.0, error=str(e)))

    return BatchExtractResponse(schema_name=schema_name, results=results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
