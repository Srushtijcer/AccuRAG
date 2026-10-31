# AccuRAG — College Major Project Viva Preparation & Demonstration Guide

## 1. Project Elevator Pitch (30-second summary)
> *"AccuRAG is a verified Retrieval-Augmented Generation chatbot designed to solve the critical problem of AI hallucinations in academic and technical domains. Unlike basic RAG chatbots that blindly trust language model outputs, AccuRAG operates under the principle of **'Not Just Answers. Verified Intelligence'**. It retrieves information from uploaded course documents and live web sources, extracts high-resolution visual diagram evidence from PDFs, and runs an automated Self-Verification engine that breaks answers into claims, checks them against evidence, and computes an honest Trust Score."*

---

## 2. Key Questions the Evaluator May Ask & How to Answer

### Q1: What makes AccuRAG different from a standard RAG system?
- **Answer**: Standard RAG simply embeds text, does cosine similarity retrieval, and feeds it into an LLM. It suffers from hallucinations, cannot handle visual diagrams, and never verifies whether the generated text actually matches the context.
- **AccuRAG's four key innovations**:
  1. **Self-Verification Engine**: Extracts atomic factual claims from the generated answer and checks each claim for evidence support or contradiction.
  2. **Visual Evidence Retrieval**: Uses PyMuPDF to locate the exact page containing diagrams/figures (inspecting neighboring pages if a figure title continues) and renders high-res PNG proofs for the user.
  3. **Hybrid Smart Mode**: Intelligently routes queries between uploaded course notes and live web search, avoiding fabricated information.
  4. **Smart Query Decomposition**: Breaks multi-part compound queries into structured sub-tasks.

### Q2: What embedding model and LLM are used, and why?
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors). It provides fast, accurate semantic similarity on CPU without requiring expensive GPU clusters.
- **LLM**: `google/flan-t5-base` (248M parameter encoder-decoder Seq2Seq model). It runs locally, respects strict context prompts, and avoids unauthorized cloud data transfers.

### Q3: How does the Self-Verification and Self-Correction module work?
1. **Claim Extraction**: The answer is parsed into propositional claims, stripping conversational fillers.
2. **Grounding Analysis**: Token overlap (Jaccard similarity) and negation contradiction checks are evaluated against retrieved document chunks and web snippets.
3. **Status Classification**: Each claim is marked `supported`, `partially_supported`, `contradicted`, or `unsupported`.
4. **Self-Correction**: If unsupported claims exist, the system prunes them and re-grounds the answer strictly in verified evidence passages.
5. **Trust Score Calculation**:
   $$\text{Trust Score} = \left(0.35 \times \text{Relevance} + 0.40 \times \text{Claim Support Ratio} + 0.25 \times \text{Coverage} - \text{Penalty}\right) \times 100\%$$

### Q4: How does Diagram Retrieval work with PyMuPDF?
- **Detection**: The query is scanned for visual intent (`diagram`, `figure`, `model`, `flowchart`).
- **Page Locator**: In textbooks, a diagram is often introduced on one page (e.g., Page 3: *"A symmetric encryption scheme has five ingredients (Figure 2.1)"*) while the actual figure graphic is printed on the top of the next page (Page 4: *"Figure 2.1. Simplified Model of Conventional Encryption"*).
- **Neighboring Page Inspection**: AccuRAG scans `best_page`, `best_page + 1`, and `best_page - 1` for figure headers and drawing objects, identifying the true diagram page and rendering it at `Matrix(2.0, 2.0)` resolution.

---

## 3. Step-by-Step Live Demonstration Checklist

### Demo Step 1: Document Upload & Hybrid Vector Indexing
1. Open the UI at `http://localhost:5173`.
2. Notice the existing indexed documents in the **Documents** tab (`BCS703_CNS_Module1.pdf`).
3. Click **+ Upload PDF** and upload another document to show real-time chunking and indexing.

### Demo Step 2: Diagram Retrieval with Neighboring Page Proof
1. Type question:  
   `"Explain the Symmetric Cipher Model with a neat diagram."`
2. Select **Smart Mode** and **Detailed Mode**.
3. Click Send (`↑`).
4. **Show the Evaluator**:
   - The structured 5-component explanation (Plaintext, Encryption algorithm, Secret key, Ciphertext, Decryption algorithm).
   - The **Visual Evidence Proof** card showing the rendered Page 4 diagram.
   - Click **Enlarge Fullscreen** to show the high-res Lightbox modal.
   - Expand the **Evidence Claim Breakdown** to show how each component claim was verified.

### Demo Step 3: Current Web Search & Live Citation
1. Type question:  
   `"What are the latest developments in retrieval-augmented generation?"`
2. Switch to **Web Mode** (or leave on **Smart Mode**).
3. Click Send.
4. **Show the Evaluator**:
   - Live web citations with clickable links and real URLs.
   - Domain attribution badges (`arxiv.org`, `wikipedia.org`, etc.).
   - Transparent Trust Score computed on web snippets.

### Demo Step 4: Complex Multi-Part Query Decomposition
1. Type question:  
   `"Explain the symmetric cipher model, describe its five components, and compare it with public-key encryption."`
2. Click Send.
3. **Show the Evaluator**:
   - The **Smart Query Decomposition** banner showing 3 discrete sub-tasks:
     1. Explain the symmetric cipher model
     2. Describe its five components
     3. Compare it with public-key encryption
   - Unified answer synthesized from multiple document passages.

### Demo Step 5: Handling Out-of-Document Queries (Zero Hallucination)
1. In **Document Mode**, type:  
   `"Who is the current Prime Minister of Australia?"`
2. Click Send.
3. **Show the Evaluator**:
   - AccuRAG honestly responds:  
     *"The available documents and search sources do not contain enough verified information to answer this question."*
   - Status badge displays **"Insufficient evidence"** with low trust score.
   - Proves AccuRAG does not hallucinate answers when evidence is missing.
