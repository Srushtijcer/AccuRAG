# AccuRAG — Not Just Answers. Verified Intelligence.

> **Final-Year Engineering Major Project**  
> A Document-Grounded Retrieval-Augmented Generation (RAG) System featuring **Hybrid Document & Web Retrieval**, **Visual Diagram Evidence Extraction**, **Automated Self-Verification & Self-Correction**, and **Smart Query Decomposition**.

---

## 1. System Architecture

```mermaid
flowchart TD
    User([User / Browser]) <--> ReactUI[React 19 Frontend :5173]
    ReactUI <--> FastAPI[FastAPI Backend :8000]

    subgraph Backend Engine
        FastAPI --> DocEngine[Document Engine]
        FastAPI --> WebEngine[Web Engine]
        FastAPI --> DecompEngine[Decomposition Engine]
        FastAPI --> GenEngine[Flan-T5 LLM Generator]
        FastAPI --> VerifEngine[Self-Verification Engine]
        
        DocEngine --> PyMuPDF[PyMuPDF Page Parser & Visual Renderer]
        DocEngine --> EmbedModel[all-MiniLM-L6-v2 Embeddings]
        DocEngine --> VectorStore[(In-Memory Vector Store + RRF)]

        WebEngine --> DDG[DuckDuckGo Free Lite Search]
        WebEngine --> TavilySerper[Tavily / Serper API Optional]

        VerifEngine --> ClaimExtractor[Claim Deconstruction]
        VerifEngine --> GroundingCheck[Token Overlap & Negation Checker]
        VerifEngine --> TrustScorer[Transparent Trust Scorer]
        VerifEngine --> Corrector[Self-Correction Loop]
    end

    DocEngine --> ImgServer[/diagrams/ Static High-Res PNGs]
    ImgServer -.-> ReactUI
```

---

## 2. Key Capabilities & Engineering Highlights

1. **Hybrid Document & Web Retrieval**:
   - **Document Mode**: Queries course PDF notes (`BCS703_CNS_Module1.pdf`, etc.) using dense vector embeddings + lexical keyword matching (RRF).
   - **Web Mode**: Queries live web sources using a zero-config DuckDuckGo Lite integration (no paid API key required).
   - **Smart Mode**: Automatically determines whether to query documents, web sources, or both.
2. **Visual Evidence & Diagram Retrieval**:
   - Detects visual requests (`diagram`, `figure`, `model`, `flowchart`).
   - Scans neighboring pages for figure titles (e.g., text on Page 3 citing `Figure 2.1` located on Page 4).
   - Renders 2.0x high-resolution page captures for instant inspection and lightbox zoom.
3. **Automated Self-Verification & Self-Correction**:
   - Decomposes generated answers into atomic factual claims.
   - Evaluates each claim against retrieved evidence (`Supported by evidence`, `Partially supported`, `Conflicting evidence`, `Insufficient evidence`).
   - Automatically corrects or qualifies unsupported statements (up to 2 iterations).
   - Computes a transparent, measurable **Trust Score** ($0-100\%$).
4. **Smart Query Decomposition**:
   - Breaks complex multi-part questions into manageable sub-tasks.
   - Gathers evidence for each sub-question and synthesizes a unified response.

---

## 3. Technology Stack

- **Frontend**: React 19, Vite 8, Lucide React, Axios.
- **Backend API**: Python 3.14 / FastAPI, Uvicorn, Pydantic v2.
- **Vector Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Language Model**: HuggingFace Transformers (`google/flan-t5-base`).
- **PDF & Visual Engine**: PyMuPDF (`fitz`), PyPDF.
- **Web Retrieval**: HTTPX, LXML, DuckDuckGo Lite scraper (with optional Tavily/Serper support).

---

## 4. Installation & Execution Commands

### Prerequisites
- Python 3.10+ (Current: Python 3.14)
- Node.js 18+ and npm

### Terminal 1: Start the FastAPI Backend
```powershell
cd c:\Users\srush\Desktop\AccuRAG\backend
# Activate virtual environment
.\venv\Scripts\Activate.ps1
# Start Uvicorn server on port 8000
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
*API Swagger Documentation will be live at: `http://127.0.0.1:8000/docs`*

### Terminal 2: Start the React Frontend
```powershell
cd c:\Users\srush\Desktop\AccuRAG\frontend
npm run dev
```
*Frontend application will be accessible at: `http://localhost:5173`*

---

## 5. Automated Test Suite

AccuRAG includes a 15-scenario automated verification suite in `backend/test_accurag_suite.py`:

```powershell
cd c:\Users\srush\Desktop\AccuRAG\backend
.\venv\Scripts\python.exe -m unittest test_accurag_suite.py -v
```

### Verified Test Scenarios:
1. `test_01_upload_valid_pdf`: Upload and index valid PDF.
2. `test_02_upload_invalid_or_empty_file`: Reject 0-byte, corrupt, or non-PDF files.
3. `test_03_in_document_question`: Grounded answer with source citations.
4. `test_04_out_of_document_question`: Honest limitation statement when evidence is absent.
5. `test_05_multi_evidence_question`: Multi-passage aggregation for comprehensive answers.
6. `test_06_web_mode_query`: Live web search with real URLs and snippets.
7. `test_07_smart_mode_hybrid`: Automatic intent routing between docs and web.
8. `test_08_unsupported_claim_detection`: Flagging unsupported claims in verification module.
9. `test_09_conflicting_source_detection`: Detecting cross-source disagreements.
10. `test_10_diagram_retrieval`: Locating figures across neighboring pages.
11. `test_11_verify_page_numbers_and_image_serving`: Serving high-res PNG visual proof.
12. `test_12_empty_question_and_error_handling`: Validation error handling (400/422).
13. `test_13_query_decomposition`: Deconstructing complex compound questions into sub-tasks.
14. `test_14_missing_web_credentials_resilience`: 100% offline document RAG independence.
15. `test_15_persistence_and_health_check`: Verifying workspace persistence and health.

---

## 6. Environment Variables (Optional)

The application works completely out-of-the-box with zero paid API keys. If you wish to use external search providers, you can set:
```bash
TAVILY_API_KEY="tvly-..."
SERPER_API_KEY="serp-..."
```
