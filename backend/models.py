from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PreviousContext(BaseModel):
    previous_question: Optional[str] = None
    previous_answer: Optional[str] = None
    previous_sources: Optional[List[Dict[str, Any]]] = None
    previous_web_sources: Optional[List[Dict[str, Any]]] = None
    previous_diagrams: Optional[List[Dict[str, Any]]] = None
    previous_mode: Optional[str] = None
    previous_page: Optional[int] = None
    previous_format: Optional[str] = None

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question text to query")
    mode: str = Field("smart", description="Answer mode: 'document', 'web', or 'smart'")
    explanation_mode: str = Field("detailed", description="Explanation style: 'easy', 'detailed', or 'exam'")
    answer_length: Optional[str] = Field("medium", description="Length: 'short', 'medium', or 'detailed'")
    exam_marks: Optional[int] = Field(None, description="Exam marks: 2, 5, 10, or 15")
    conversation_id: Optional[str] = Field(None, description="Unique conversation session ID")
    previous_context: Optional[PreviousContext] = None

class SourceChunk(BaseModel):
    document: str
    page: int
    text: str
    similarity: float

class WebSource(BaseModel):
    title: str
    url: str
    snippet: str
    domain: str

class DiagramInfo(BaseModel):
    page: int
    filename: str
    path: str
    caption: Optional[str] = None
    document: Optional[str] = None
    source_type: Optional[str] = "document"  # "document" or "web"
    attribution: Optional[str] = None

class ClaimVerification(BaseModel):
    claim: str
    status: str  # "supported", "partially_supported", "contradicted", "unsupported", "company_claim", "unconfirmed_prediction"
    source: Optional[str] = None
    source_type: Optional[str] = None
    confidence: float = 1.0
    claim_type: Optional[str] = "fact"  # "confirmed_fact", "company_claim", "prediction", "feature_claim"

class VerificationResult(BaseModel):
    status: str  # "Supported by document evidence", "Supported by web evidence", "Supported by both", "Partially supported", "Conflicting evidence", "Insufficient evidence", "Verification unavailable"
    trust_score: float
    self_corrected: bool = False
    claims: List[ClaimVerification] = []
    contradictions: List[str] = []
    notes: Optional[str] = None

class DecompositionStep(BaseModel):
    sub_question: str
    evidence_count: int
    summary: str

class SectionItem(BaseModel):
    heading: str
    content: str

class AskResponse(BaseModel):
    question: str
    answer: str
    sections: Optional[List[SectionItem]] = None
    answer_format: Optional[str] = None
    answer_length: Optional[str] = None
    mode: str
    explanation_mode: str
    page: Optional[int] = None
    sources: List[SourceChunk] = []
    web_sources: List[WebSource] = []
    diagrams: List[DiagramInfo] = []
    verification: VerificationResult
    verification_status: Optional[str] = None
    evidence_coverage: Optional[str] = None  # "Strong evidence coverage", "Partial evidence coverage", "Limited evidence coverage"
    decomposition: Optional[List[DecompositionStep]] = None
    warnings: Optional[List[str]] = None
    conversation_id: Optional[str] = None
    trust_score: float

class UploadResponse(BaseModel):
    filename: str
    pages: int
    text_length: int
    chunks_created: int
    diagrams_found: str
    message: str
    ocr_notice: Optional[str] = None
