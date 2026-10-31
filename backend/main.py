import os
import sys
import re
import uuid
from typing import List, Optional, Dict, Any

# Ensure UTF-8 output encoding for console on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Enable offline loading for HuggingFace if models are cached
if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"
if "TRANSFORMERS_OFFLINE" not in os.environ:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from models import (
    AskRequest,
    AskResponse,
    PreviousContext,
    UploadResponse,
    SourceChunk,
    WebSource,
    DiagramInfo,
    VerificationResult,
    DecompositionStep,
    ClaimVerification,
    SectionItem
)
from document_engine import DocumentEngine
from web_engine import WebEngine
from verification_engine import VerificationEngine
from generation_engine import GenerationEngine
from decomposition_engine import DecompositionEngine

# =========================================================
# APP INITIALIZATION
# =========================================================

app = FastAPI(
    title="AccuRAG API",
    description="AccuRAG — Not Just Answers. Verified Intelligence. Document-grounded RAG with Hybrid Web Search, Visual Evidence, and Self-Verification.",
    version="2.1.0"
)

# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# DIRECTORIES & ENGINES
# =========================================================

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
DIAGRAM_DIR = os.path.join(os.path.dirname(__file__), "diagrams")
os.makedirs(UPLOAD_DIR, exist_ok=True)
DIAGRAM_DIR_EXISTS = os.makedirs(DIAGRAM_DIR, exist_ok=True)

# Mount diagrams for static serving with path validation
app.mount("/diagrams", StaticFiles(directory=DIAGRAM_DIR), name="diagrams")

# Instantiate Core Engines
doc_engine = DocumentEngine(upload_dir=UPLOAD_DIR, diagram_dir=DIAGRAM_DIR)
web_engine = WebEngine()
verification_engine = VerificationEngine(embed_model=doc_engine.embed_model)
gen_engine = GenerationEngine()
decomp_engine = DecompositionEngine()

# In-memory Conversation Session Store isolated per conversation/session
CONVERSATION_SESSIONS: Dict[str, Dict[str, Any]] = {}

# Automatically index existing PDFs on startup if available (deduplicating identical files)
def _auto_index_existing_uploads():
    try:
        pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
        seen_sizes = {}
        for pdf_name in sorted(pdf_files):
            file_path = os.path.join(UPLOAD_DIR, pdf_name)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                # Skip duplicate files with identical size
                if file_size in seen_sizes:
                    print(f"[Startup] Skipping duplicate upload '{pdf_name}' (identical size to '{seen_sizes[file_size]}').")
                    continue
                seen_sizes[file_size] = pdf_name
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                print(f"[Startup] Indexing uploaded file: {pdf_name} ({len(file_bytes)} bytes)...")
                doc_engine.process_pdf(pdf_name, file_bytes, replace_existing=False)
        print(f"[Startup] Ready with {len(doc_engine.documents)} documents and {len(doc_engine.chunks)} indexed chunks.")
    except Exception as e:
        print(f"[Startup] Error during auto-indexing: {e}")

_auto_index_existing_uploads()

# =========================================================
# HEALTH / SYSTEM STATUS ENDPOINT
# =========================================================

@app.get("/")
def home():
    web_status = web_engine.is_configured()
    return {
        "status": "online",
        "message": "AccuRAG Backend is running!",
        "version": "2.1.0",
        "documents_indexed": len(doc_engine.documents),
        "total_chunks": len(doc_engine.chunks),
        "web_search": web_status,
        "features": [
            "Dense + Lexical Hybrid Document Retrieval",
            "DuckDuckGo / Tavily Hybrid Web Retrieval",
            "Intelligent Format Selection & Follow-up Transformations",
            "Self-Verification & Fact-Checking Engine",
            "Self-Correction & Hallucination Mitigation",
            "Smart Query Decomposition",
            "Visual Evidence & Diagram Locator with Neighboring Page Inspection",
            "Transparent Trust Scoring & Evidence Coverage"
        ]
    }

# =========================================================
# DOCUMENT MANAGEMENT ENDPOINTS
# =========================================================

@app.get("/documents")
def list_documents():
    """List all currently indexed documents in AccuRAG workspace."""
    docs = []
    for name, meta in doc_engine.documents.items():
        docs.append({
            "filename": meta["filename"],
            "pages": meta["pages"],
            "text_length": meta["text_length"],
            "chunks_count": meta["chunks_count"],
            "ocr_notice": meta.get("ocr_notice")
        })
    return {"documents": docs, "total_chunks": len(doc_engine.chunks)}

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and index a PDF into the AccuRAG vector store."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    file_bytes = await file.read()
    res = doc_engine.process_pdf(file.filename, file_bytes, replace_existing=True)

    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Failed to process PDF."))

    return UploadResponse(
        filename=res["filename"],
        pages=res["pages"],
        text_length=res["text_length"],
        chunks_created=res["chunks_created"],
        diagrams_found="Rendered on-demand when relevant visual evidence is requested",
        message=res["message"],
        ocr_notice=res.get("ocr_notice")
    )

@app.delete("/conversations/{conversation_id}")
def clear_conversation(conversation_id: str):
    """Clear conversation history and session memory for a given conversation."""
    if conversation_id in CONVERSATION_SESSIONS:
        del CONVERSATION_SESSIONS[conversation_id]
        return {"status": "cleared", "conversation_id": conversation_id}
    return {"status": "not_found", "conversation_id": conversation_id}

# =========================================================
# CORE RAG QUERY EXECUTION LOGIC
# =========================================================

def _execute_accurag_pipeline(
    question: str,
    mode: str = "smart",
    explanation_mode: str = "detailed",
    answer_length: str = "medium",
    exam_marks: Optional[int] = None,
    conversation_id: Optional[str] = None,
    previous_context: Optional[PreviousContext] = None
) -> Dict[str, Any]:
    """Execute complete RAG pipeline: Conversational Transform -> Intent Routing -> Retrieval -> Generation -> Verification."""
    clean_q = question.strip()
    mode = mode.lower()
    explanation_mode = explanation_mode.lower()
    conv_id = conversation_id or str(uuid.uuid4())
    warnings = []

    if not clean_q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # ---------------------------------------------------------
    # 1. CONVERSATIONAL FOLLOW-UP & TRANSFORMATION CHECK
    # ---------------------------------------------------------
    session = CONVERSATION_SESSIONS.get(conv_id)
    prev_q = None
    prev_a = None
    prev_doc_res = []
    prev_web_res = []
    prev_diag = []
    prev_diag_page = None
    prev_mode = mode

    if previous_context and previous_context.previous_answer:
        prev_q = previous_context.previous_question
        prev_a = previous_context.previous_answer
        prev_doc_res = previous_context.previous_sources or []
        prev_web_res = previous_context.previous_web_sources or []
        prev_diag = previous_context.previous_diagrams or []
        prev_mode = previous_context.previous_mode or mode
        prev_diag_page = previous_context.previous_page
    elif session:
        prev_q = session.get("last_question")
        prev_a = session.get("last_answer")
        prev_doc_res = session.get("last_doc_results", [])
        prev_web_res = session.get("last_web_results", [])
        prev_diag = session.get("last_diagrams", [])
        prev_mode = session.get("last_mode", mode)
        prev_diag_page = session.get("last_page")

    # Detect follow-up formatting intent (e.g. "give in points", "in simple words", "make it shorter", etc.)
    followup_intent = gen_engine.classify_followup_intent(clean_q)
    if followup_intent:
        if prev_a:
            # Transform previous answer without re-retrieval
            diag_info = prev_diag[0] if prev_diag else None
            transformed_ans, ans_format, ans_len = gen_engine.transform_answer(
                intent=followup_intent,
                previous_answer=prev_a,
                previous_question=prev_q or clean_q,
                diagram_info=diag_info,
                exam_marks=exam_marks,
                answer_length=answer_length
            )
            sections = gen_engine.extract_sections(transformed_ans)
            
            # Verify transformed answer against original evidence
            verif = verification_engine.verify_answer(clean_q, transformed_ans, prev_doc_res, prev_web_res)

            # Update session
            CONVERSATION_SESSIONS[conv_id] = {
                "last_question": clean_q,
                "last_answer": transformed_ans,
                "last_mode": prev_mode,
                "last_doc_results": prev_doc_res,
                "last_web_results": prev_web_res,
                "last_diagrams": prev_diag,
                "last_page": prev_diag_page
            }

            return {
                "question": clean_q,
                "answer": transformed_ans,
                "sections": sections,
                "answer_format": ans_format,
                "answer_length": ans_len,
                "mode": prev_mode,
                "explanation_mode": explanation_mode,
                "page": prev_diag_page,
                "sources": prev_doc_res,
                "web_sources": prev_web_res,
                "diagrams": prev_diag,
                "verification": verif,
                "verification_status": verif["status"],
                "evidence_coverage": verif.get("evidence_coverage", "Strong evidence coverage"),
                "decomposition": None,
                "warnings": None,
                "conversation_id": conv_id,
                "trust_score": verif["trust_score"]
            }
        else:
            # Prompt user to ask an initial question before applying format transformations
            clarification = f"I notice you requested a format modification ('{clean_q}'), but no previous question has been asked in this session yet. Please ask an initial question or provide a topic first, and I will gladly provide the explanation in your desired format!"
            return {
                "question": clean_q,
                "answer": clarification,
                "sections": [{"heading": "Clarification Required", "content": clarification}],
                "answer_format": "clarification",
                "answer_length": "short",
                "mode": mode,
                "explanation_mode": explanation_mode,
                "page": None,
                "sources": [],
                "web_sources": [],
                "diagrams": [],
                "verification": {
                    "status": "Verification unavailable",
                    "trust_score": 50.0,
                    "evidence_coverage": "Limited evidence coverage",
                    "self_corrected": False,
                    "claims": [],
                    "contradictions": [],
                    "notes": "Awaiting initial query from user."
                },
                "verification_status": "Verification unavailable",
                "evidence_coverage": "Limited evidence coverage",
                "decomposition": None,
                "warnings": None,
                "conversation_id": conv_id,
                "trust_score": 50.0
            }

    # ---------------------------------------------------------
    # 2. INTENT ROUTING (DOCUMENT, WEB, OR SMART)
    # ---------------------------------------------------------
    effective_mode = mode
    web_needed = False
    doc_needed = False
    has_documents = len(doc_engine.chunks) > 0

    if mode == "document":
        doc_needed = True
        web_needed = False
    elif mode == "web":
        doc_needed = False
        web_needed = True
    else:  # "smart" mode
        effective_mode = "smart"
        current_info_keywords = ["latest", "recent", "developments", "current", "news", "trend", "today", "updates", "2024", "2025", "2026", "frontiers"]
        comparison_keywords = ["compare", "difference between", "versus", "vs", "both in notes and web", "compare my uploaded notes"]
        note_keywords = ["notes", "document", "uploaded", "syllabus", "module", "lecture", "textbook", "in my pdf", "in the pdf"]

        is_current = any(k in clean_q.lower() for k in current_info_keywords)
        is_comparison = any(k in clean_q.lower() for k in comparison_keywords)
        is_explicit_doc = any(k in clean_q.lower() for k in note_keywords)

        if is_comparison:
            doc_needed = has_documents
            web_needed = True
        elif is_current and not is_explicit_doc:
            web_needed = True
            doc_needed = False
        elif is_explicit_doc:
            doc_needed = has_documents
            web_needed = False
        else:
            doc_needed = has_documents
            web_needed = not has_documents

    # ---------------------------------------------------------
    # 3. QUERY DECOMPOSITION
    # ---------------------------------------------------------
    decomposition_steps = None
    sub_questions = [clean_q]
    if decomp_engine.is_complex_query(clean_q):
        sub_questions = decomp_engine.decompose(clean_q)
        if len(sub_questions) > 1:
            decomposition_steps = []

    # ---------------------------------------------------------
    # 4. DOCUMENT RETRIEVAL
    # ---------------------------------------------------------
    doc_results = []
    if doc_needed and has_documents:
        seen_texts = set()
        for sub_q in sub_questions:
            sub_matches = doc_engine.retrieve(sub_q, top_k=4)
            for m in sub_matches:
                if m["text"] not in seen_texts:
                    seen_texts.add(m["text"])
                    doc_results.append(m)

        doc_results.sort(key=lambda x: x.get("hybrid_score", x.get("similarity", 0)), reverse=True)
        doc_results = doc_results[:5]

        # Smart mode fallback to web if document retrieval returned no matches or very low similarity
        if mode == "smart" and (not doc_results or max([d.get("similarity", 0) for d in doc_results], default=0) < 0.28):
            web_needed = True

    # ---------------------------------------------------------
    # 5. WEB SEARCH RETRIEVAL
    # ---------------------------------------------------------
    web_results = []
    if web_needed:
        web_results, search_warning = web_engine.search_with_diagnostics(clean_q, max_results=4)
        if search_warning:
            warnings.append(search_warning)

        if not web_results and web_needed and mode == "web":
            warnings.append("Web search returned no external results. Please verify your internet connection or search terms.")

    # ---------------------------------------------------------
    # 6. INSUFFICIENT EVIDENCE CHECK
    # ---------------------------------------------------------
    if not doc_results and not web_results:
        limitation_text = (
            "The uploaded documents do not contain sufficient verified information to answer this question. "
            "Please upload a relevant document or switch to Web Search Mode."
            if mode == "document" else
            "The available documents and search sources do not contain enough verified information to answer this question. Please upload a relevant document or verify your query."
        )
        return {
            "question": clean_q,
            "answer": limitation_text,
            "sections": [{"heading": "Notice", "content": limitation_text}],
            "answer_format": "insufficient",
            "answer_length": "short",
            "mode": effective_mode,
            "explanation_mode": explanation_mode,
            "page": None,
            "sources": [],
            "web_sources": [],
            "diagrams": [],
            "verification": {
                "status": "Insufficient evidence",
                "trust_score": 15.0,
                "evidence_coverage": "Limited evidence coverage",
                "self_corrected": False,
                "claims": [],
                "contradictions": [],
                "notes": "No matching document chunks or web evidence could be retrieved."
            },
            "verification_status": "Insufficient evidence",
            "evidence_coverage": "Limited evidence coverage",
            "decomposition": None,
            "warnings": warnings if warnings else None,
            "conversation_id": conv_id,
            "trust_score": 15.0
        }

    # ---------------------------------------------------------
    # 7. DIAGRAM / PHOTO LOCATOR & CONTEXT EXPANSION
    # ---------------------------------------------------------
    diagrams_found = []
    best_page = doc_results[0]["page"] if doc_results else None
    primary_doc = doc_results[0]["document"] if doc_results else None
    diagram_info = None

    # Check if a visual diagram or genuine photo is requested
    photo_requested = bool(re.search(r"\b(photo|photograph|picture|image|diagram|figure)\b", clean_q, re.IGNORECASE))

    # A. Check document visual evidence first if document results are present
    if doc_engine.is_diagram_requested(clean_q) and primary_doc and best_page:
        diagram_page = doc_engine.locate_diagram_page(clean_q, best_page, primary_doc)
        rendered = doc_engine.render_page_diagram(primary_doc, diagram_page)
        if rendered:
            diagram_info = {
                "page": diagram_page,
                "filename": rendered["filename"],
                "path": rendered["path"],
                "document": primary_doc,
                "caption": f"Verified visual evidence from {primary_doc} (Page {diagram_page})",
                "source_type": "document"
            }
            diagrams_found.append(diagram_info)
            best_page = diagram_page

            # Expand document context with neighboring pages around the diagram
            neighbor_chunks = doc_engine.get_surrounding_page_chunks(primary_doc, diagram_page, span=1)
            seen_texts = set(d["text"] for d in doc_results)
            for nc in neighbor_chunks:
                if nc["text"] not in seen_texts:
                    seen_texts.add(nc["text"])
                    doc_results.append(nc)

    elif primary_doc and best_page and mode == "document":
        # Render source page capture for visual proof
        rendered = doc_engine.render_page_diagram(primary_doc, best_page)
        if rendered:
            diagrams_found.append({
                "page": best_page,
                "filename": rendered["filename"],
                "path": rendered["path"],
                "document": primary_doc,
                "caption": f"Source page capture for {primary_doc} (Page {best_page})",
                "source_type": "document"
            })

    # B. If photo/image was requested and we are in web mode or smart mode without doc diagram, search web photos
    if (photo_requested or mode == "web") and not diagrams_found and web_results:
        web_photos = web_engine.search_images(clean_q, max_results=1)
        if web_photos:
            diagrams_found.extend(web_photos)
            diagram_info = web_photos[0]

    # ---------------------------------------------------------
    # 8. BUILD SYNTHESIZED CONTEXT
    # ---------------------------------------------------------
    context_blocks = []
    if doc_results:
        context_blocks.append("--- DOCUMENT EVIDENCE ---")
        for idx, d in enumerate(doc_results, start=1):
            context_blocks.append(f"[{idx}] Source: {d['document']} (Page {d['page']}):\n{d['text']}")

    if web_results:
        context_blocks.append("--- WEB SEARCH EVIDENCE ---")
        for idx, w in enumerate(web_results, start=1):
            context_blocks.append(f"[Web {idx}] {w['title']} ({w['domain']}):\n{w['snippet']}")

    context_str = "\n\n".join(context_blocks)

    # ---------------------------------------------------------
    # 9. GENERATE COMPLETE STRUCTURED ANSWER
    # ---------------------------------------------------------
    raw_answer, sections, ans_format, ans_len = gen_engine.generate(
        question=clean_q,
        context=context_str,
        explanation_mode=explanation_mode,
        diagram_info=diagram_info,
        mode=effective_mode,
        web_sources=web_results,
        doc_sources=doc_results,
        answer_length=answer_length,
        exam_marks=exam_marks
    )

    # Record decomposition step details if decomposition was executed
    if decomposition_steps is not None:
        for sq in sub_questions:
            sq_matches = [d for d in doc_results if any(w in d['text'].lower() for w in sq.lower().split() if len(w) > 3)]
            decomposition_steps.append({
                "sub_question": sq,
                "evidence_count": len(sq_matches),
                "summary": f"Retrieved {len(sq_matches)} relevant source evidence passages."
            })

    # ---------------------------------------------------------
    # 10. SELF-VERIFICATION & SELF-CORRECTION
    # ---------------------------------------------------------
    initial_verif = verification_engine.verify_answer(clean_q, raw_answer, doc_results, web_results)
    final_answer, final_verif, was_corrected = verification_engine.self_correct_if_needed(
        clean_q, raw_answer, initial_verif, doc_results, web_results, max_attempts=2
    )

    # Re-extract sections if self-correction added notes
    if was_corrected:
        sections = gen_engine.extract_sections(final_answer)

    # ---------------------------------------------------------
    # 11. PERSIST SESSION MEMORY
    # ---------------------------------------------------------
    CONVERSATION_SESSIONS[conv_id] = {
        "last_question": clean_q,
        "last_answer": final_answer,
        "last_mode": effective_mode,
        "last_doc_results": [
            {
                "document": d["document"],
                "page": d["page"],
                "text": d["text"],
                "similarity": d.get("similarity", 0.85)
            }
            for d in doc_results
        ],
        "last_web_results": [
            {
                "title": w["title"],
                "url": w["url"],
                "snippet": w["snippet"],
                "domain": w["domain"]
            }
            for w in web_results
        ],
        "last_diagrams": diagrams_found,
        "last_page": best_page
    }

    return {
        "question": clean_q,
        "answer": final_answer,
        "sections": sections,
        "answer_format": ans_format,
        "answer_length": ans_len,
        "mode": effective_mode,
        "explanation_mode": explanation_mode,
        "page": best_page,
        "sources": [
            {
                "document": d["document"],
                "page": d["page"],
                "text": d["text"],
                "similarity": d.get("similarity", 0.85)
            }
            for d in doc_results
        ],
        "web_sources": [
            {
                "title": w["title"],
                "url": w["url"],
                "snippet": w["snippet"],
                "domain": w["domain"]
            }
            for w in web_results
        ],
        "diagrams": diagrams_found,
        "verification": final_verif,
        "verification_status": final_verif["status"],
        "evidence_coverage": final_verif.get("evidence_coverage", "Strong evidence coverage"),
        "decomposition": decomposition_steps,
        "warnings": warnings if warnings else None,
        "conversation_id": conv_id,
        "trust_score": final_verif["trust_score"]
    }

# =========================================================
# ASKING / QUERY ENDPOINTS
# =========================================================

@app.get("/ask")
def ask_get(
    question: str = Query(..., description="Query text"),
    mode: str = Query("smart", description="Mode: 'document', 'web', or 'smart'"),
    explanation_mode: str = Query("detailed", description="Style: 'easy', 'detailed', or 'exam'"),
    answer_length: str = Query("medium", description="Length: 'short', 'medium', or 'detailed'"),
    exam_marks: Optional[int] = Query(None, description="Exam marks: 2, 5, 10, or 15"),
    conversation_id: Optional[str] = Query(None, description="Conversation session ID")
):
    """GET endpoint compatible with existing frontend queries."""
    result = _execute_accurag_pipeline(
        question=question,
        mode=mode,
        explanation_mode=explanation_mode,
        answer_length=answer_length,
        exam_marks=exam_marks,
        conversation_id=conversation_id
    )
    return result
@app.post("/ask", response_model=AskResponse)
def ask_post(request: AskRequest):
    """POST endpoint with structured Pydantic validation and conversational context."""
    result = _execute_accurag_pipeline(
        question=request.question,
        mode=request.mode,
        explanation_mode=request.explanation_mode,
        answer_length=request.answer_length or "medium",
        exam_marks=request.exam_marks,
        conversation_id=request.conversation_id,
        previous_context=request.previous_context
    )
    return AskResponse(**result)

# =========================================================
# STANDALONE VERIFICATION & DECOMPOSITION (FOR DEMOS / VIVA)
# =========================================================

@app.post("/verify")
def verify_claim_endpoint(payload: Dict[str, Any]):
    """Demonstrate claim verification independently for project viva."""
    text = payload.get("text", "")
    sources = payload.get("sources", [])
    if not text:
        raise HTTPException(status_code=400, detail="Missing text to verify.")
    claims = verification_engine.extract_claims(text)
    verified = [
        verification_engine.verify_single_claim(c, sources, [])
        for c in claims
    ]
    return {
        "claims_extracted": claims,
        "verification_results": verified,
        "total": len(claims)
    }

@app.post("/decompose")
def decompose_query_endpoint(payload: Dict[str, Any]):
    """Demonstrate query decomposition independently for project viva."""
    query = payload.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="Missing query to decompose.")
    is_complex = decomp_engine.is_complex_query(query)
    sub_tasks = decomp_engine.decompose(query)
    return {
        "query": query,
        "is_complex": is_complex,
        "sub_tasks": sub_tasks
    }
