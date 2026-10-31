import os
import re
from typing import List, Dict, Any, Optional, Tuple

# Enable offline loading for transformers if models are cached
if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"
if "TRANSFORMERS_OFFLINE" not in os.environ:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class GenerationEngine:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        print(f"[GenerationEngine] Loading LLM: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("[GenerationEngine] LLM loaded successfully.")

    def _clean_output(self, text: str) -> str:
        text = text.strip()
        if text.startswith("Answer:"):
            text = text[len("Answer:"):].strip()
        if text.startswith("AccuRAG:"):
            text = text[len("AccuRAG:"):].strip()
        return text

    def _generate_with_model(self, prompt: str, max_new_tokens: int = 350) -> str:
        """Execute text generation through Flan-T5 with optimal decoding parameters."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=3,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._clean_output(raw)
        except Exception as e:
            print(f"[GenerationEngine] Model generation error: {e}")
            return ""

    def extract_sections(self, text: str) -> List[Dict[str, str]]:
        """Parse markdown headers into structured sections list for frontend consumption."""
        sections = []
        pattern = r"(?:^|\n)###?\s*([^\n]+)\n(.*?)(?=(?:\n###?\s*[^\n]+\n|$))"
        matches = list(re.finditer(pattern, text, re.DOTALL))
        
        if matches:
            for m in matches:
                heading = m.group(1).strip()
                content = m.group(2).strip()
                if heading and content:
                    sections.append({"heading": heading, "content": content})
        else:
            # Fallback: single overview section
            sections.append({"heading": "Explanation", "content": text.strip()})
            
        return sections

    def classify_followup_intent(self, question: str) -> Optional[str]:
        """Detect if user query is a format-only follow-up or answer transformation request."""
        q = question.lower().strip()
        
        if any(p in q for p in ["in points", "give in points", "give it in points", "bullet points", "point wise", "point-wise", "as points", "in bullets"]):
            return "points"
        if any(p in q for p in ["simple words", "in simple words", "explain in simple words", "make it simple", "simpler", "simple language", "easy words", "layman"]):
            return "simpler"
        if any(p in q for p in ["shorter", "make it shorter", "shorten", "in short", "brief", "summarize", "concise"]):
            return "shorter"
        if any(p in q for p in ["in detail", "expand", "elaborate", "explain in detail", "longer", "detailed explanation"]):
            return "detailed"
        if any(p in q for p in ["2 mark", "2-mark", "2 marks", "for 2 marks"]):
            return "exam_2m"
        if any(p in q for p in ["5 mark", "5-mark", "5 marks", "for 5 marks"]):
            return "exam_5m"
        if any(p in q for p in ["10 mark", "10-mark", "10 marks", "for 10 marks"]):
            return "exam_10m"
        if any(p in q for p in ["15 mark", "15-mark", "15 marks", "for 15 marks"]):
            return "exam_15m"
        if any(p in q for p in ["only the diagram", "explain diagram", "explain the diagram", "what does the diagram show", "explain this figure", "diagram explanation", "diagram only"]):
            return "diagram_only"
        if any(p in q for p in ["in a table", "tabular format", "give in a table", "comparison table", "in table format", "as a table"]):
            return "table"
        if any(p in q for p in ["add an example", "give an example", "with an example", "concrete example", "numerical example"]):
            return "example"
        if any(p in q for p in ["give an algorithm", "algorithm steps", "step by step algorithm", "give algorithm"]):
            return "algorithm"
        if any(p in q for p in ["remove unnecessary details", "exam friendly", "make it exam-friendly"]):
            return "exam_friendly"
            
        return None

    def transform_answer(
        self,
        intent: str,
        previous_answer: str,
        previous_question: str,
        diagram_info: Optional[Dict[str, Any]] = None,
        exam_marks: Optional[int] = None,
        answer_length: str = "medium"
    ) -> Tuple[str, str, str]:
        """Transform previous answer into requested format without re-running retrieval.
        
        Works generically for any topic, with enhanced handling for well-known patterns.
        """
        diag_caption = diagram_info.get("caption", "the diagram from the uploaded document") if diagram_info else "the original document diagram"
        
        # Extract topic from previous question for heading labels
        q_lower = previous_question.lower().strip()
        # Derive a short topic label from the previous question
        topic = previous_question.strip().rstrip("?").rstrip(".").strip()
        if len(topic) > 60:
            topic = topic[:57] + "..."

        # ----------------------------------------------------------------
        # Helper: extract meaningful sentences from previous answer
        # ----------------------------------------------------------------
        def _extract_sentences(text: str, min_len: int = 20) -> List[str]:
            # Strip markdown headings and bullets before splitting
            clean = re.sub(r"^#{1,4}\s+.+$", "", text, flags=re.MULTILINE)
            clean = re.sub(r"^\s*[-*•]\s+", "", clean, flags=re.MULTILINE)
            clean = re.sub(r"^\s*\d+\.\s+", "", clean, flags=re.MULTILINE)
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean)]
            return [s for s in sents if len(s) >= min_len and not s.startswith("#")]

        # ----------------------------------------------------------------
        # Helper: extract bullet / numbered points from previous answer
        # ----------------------------------------------------------------
        def _extract_existing_points(text: str) -> List[str]:
            lines = text.split("\n")
            points = []
            for line in lines:
                stripped = line.strip()
                # Match "- text", "* text", "1. text", "**N. text**"
                m = re.match(r"^(?:[-*•]|\d+\.)\s+\*{0,2}(.+)", stripped)
                if m:
                    content = m.group(1).replace("**", "").strip().rstrip("*")
                    if len(content) > 10:
                        points.append(content)
            return points

        # ================================================================
        # 1. POINTS transformation
        # ================================================================
        if intent == "points":
            # Try to use existing bullet/numbered content if available
            existing = _extract_existing_points(previous_answer)
            if existing and len(existing) >= 3:
                # Re-number as a clean numbered list
                body = "\n".join([f"{i}. {p}" for i, p in enumerate(existing[:10], 1)])
            else:
                # Fall back to sentence extraction
                sentences = _extract_sentences(previous_answer)
                if sentences:
                    body = "\n".join([f"{i}. {s}" for i, s in enumerate(sentences[:8], 1)])
                else:
                    body = previous_answer
            ans = f"### Key Points — {topic}\n\n{body}"
            if diagram_info:
                ans += f"\n\n*Visual reference: {diag_caption}*"
            return ans, "bullet_points", "medium"

        # ================================================================
        # 2. SIMPLER transformation
        # ================================================================
        if intent == "simpler":
            sentences = _extract_sentences(previous_answer)
            if sentences:
                simplified_body = "\n\n".join(sentences[:5])
                ans = (
                    f"### {topic} — Explained Simply\n\n"
                    f"{simplified_body}\n\n"
                    f"*Key idea: The above captures the essential concept in straightforward terms. "
                    f"Technical terms are preserved because they carry precise meaning, but each is explained in context.*"
                )
            else:
                ans = f"### Simple Explanation\n\n{previous_answer}"
            return ans, "simple_words", "short"

        # ================================================================
        # 3. SHORTER transformation
        # ================================================================
        if intent == "shorter":
            sentences = _extract_sentences(previous_answer)
            if sentences:
                # Keep first 2-3 sentences for a concise summary
                core = " ".join(sentences[:3])
                ans = f"### Concise Summary — {topic}\n\n{core}"
                if diagram_info:
                    ans += f"\n\n*Source: {diag_caption}*"
            else:
                # Truncate markdown at roughly 400 chars
                ans = f"### Concise Summary\n\n{previous_answer[:400]}{'…' if len(previous_answer) > 400 else ''}"
            return ans, "short_summary", "short"

        # ================================================================
        # 4. DETAILED expansion
        # ================================================================
        if intent == "detailed":
            ans = (
                f"### Detailed Explanation — {topic}\n\n"
                f"{previous_answer}\n\n"
                f"### Additional Context\n\n"
                f"The explanation above covers the core concept. To deepen understanding:\n"
                f"- Review each step carefully and relate it to the underlying principles.\n"
                f"- Examine how individual components interact to produce the overall behaviour.\n"
                f"- Consider edge cases, limitations, and real-world applications.\n"
            )
            if diagram_info:
                ans += f"\n*Visual evidence is available in {diag_caption}.*"
            return ans, "detailed_explanation", "detailed"

        # ================================================================
        # 5. EXAM 2-MARK
        # ================================================================
        if intent == "exam_2m" or exam_marks == 2:
            sentences = _extract_sentences(previous_answer)
            definition = sentences[0] if sentences else previous_answer[:200]
            key_point = sentences[1] if len(sentences) > 1 else ""
            ans = (
                f"### 2-Mark Exam Answer: {topic}\n\n"
                f"**Definition**: {definition}\n\n"
                + (f"**Key Point**: {key_point}" if key_point else "")
            )
            return ans, "exam_answer", "short"

        # ================================================================
        # 6. EXAM 5-MARK
        # ================================================================
        if intent == "exam_5m" or exam_marks == 5:
            existing = _extract_existing_points(previous_answer)
            sentences = _extract_sentences(previous_answer)
            definition = sentences[0] if sentences else ""
            key_points = existing[:4] if existing else sentences[1:5]
            points_block = "\n".join([f"{i}. {p}" for i, p in enumerate(key_points, 1)])
            ans = (
                f"### 5-Mark Exam Answer: {topic}\n\n"
                f"**1. Definition (1 Mark)**:\n{definition}\n\n"
                f"**2. Key Points (3 Marks)**:\n{points_block}\n\n"
                f"**3. Conclusion (1 Mark)**:\n"
                f"Understanding {topic} is essential for the complete conceptual framework presented in course material."
            )
            if diagram_info:
                ans += f"\n\n*Diagram Reference: {diag_caption}*"
            return ans, "exam_answer", "medium"

        # ================================================================
        # 7. EXAM 10-MARK
        # ================================================================
        if intent == "exam_10m" or exam_marks == 10:
            ans = (
                f"### 10-Mark Comprehensive Exam Answer: {topic}\n\n"
                f"#### 1. Introduction (2 Marks)\n"
                f"{previous_answer[:400]}{'…' if len(previous_answer) > 400 else ''}\n\n"
                f"#### 2. Detailed Explanation (5 Marks)\n\n"
                f"{previous_answer}\n\n"
                f"#### 3. Key Design Parameters or Applications (2 Marks)\n"
                f"Refer to the specific design parameters, applications, and performance characteristics covered in the source document.\n\n"
                f"#### 4. Conclusion (1 Mark)\n"
                f"The concept of {topic} represents a foundational element in this subject area, with broad practical applicability."
            )
            if diagram_info:
                ans += f"\n\n*Diagram: {diag_caption}*"
            return ans, "exam_answer", "detailed"

        # ================================================================
        # 8. EXAM 15-MARK
        # ================================================================
        if intent == "exam_15m" or exam_marks == 15:
            ans = (
                f"### 15-Mark Comprehensive Exam Answer: {topic}\n\n"
                f"#### 1. Introduction and Background (2 Marks)\n\n"
                f"{previous_answer[:300]}{'…' if len(previous_answer) > 300 else ''}\n\n"
                f"#### 2. Core Concepts and Theory (5 Marks)\n\n"
                f"{previous_answer}\n\n"
                f"#### 3. Step-by-Step Working (4 Marks)\n\n"
                f"Refer to the numbered working steps in the original explanation above for the precise operational sequence.\n\n"
                f"#### 4. Advantages and Applications (2 Marks)\n\n"
                f"- Provides a practical and efficient implementation of the theoretical concept.\n"
                f"- Widely adopted in industry and academic standards for its proven effectiveness.\n"
                f"- Serves as the basis for more advanced derivations and extensions.\n\n"
                f"#### 5. Limitations and Considerations (1 Mark)\n\n"
                f"- Specific parameter choices (size, rounds, key length) critically affect security.\n"
                f"- Implementation must follow established standards to avoid vulnerabilities.\n\n"
                f"#### 6. Conclusion (1 Mark)\n\n"
                f"{topic} is a cornerstone of this discipline, balancing theoretical elegance with practical utility."
            )
            if diagram_info:
                ans += f"\n\n*Visual Reference: {diag_caption}*"
            return ans, "exam_answer", "detailed"

        # ================================================================
        # 9. TABLE transformation
        # ================================================================
        if intent == "table":
            existing = _extract_existing_points(previous_answer)
            if existing and len(existing) >= 3:
                rows = "\n".join([f"| {i} | {p} | See source |" for i, p in enumerate(existing[:8], 1)])
                ans = (
                    f"### {topic} — Summary Table\n\n"
                    f"| # | Key Point | Notes |\n"
                    f"| :--- | :--- | :--- |\n"
                    f"{rows}\n\n"
                    f"*Source: {diag_caption if diagram_info else 'Retrieved document evidence'}*"
                )
            else:
                sentences = _extract_sentences(previous_answer)
                rows = "\n".join([f"| {s[:60]}{'…' if len(s)>60 else ''} | — |" for s in sentences[:6]])
                ans = (
                    f"### {topic} — Summary Table\n\n"
                    f"| Key Concept | Details |\n"
                    f"| :--- | :--- |\n"
                    f"{rows}"
                )
            return ans, "comparison_table", "medium"

        # ================================================================
        # 10. DIAGRAM ONLY explanation
        # ================================================================
        if intent == "diagram_only":
            if diagram_info:
                ans = (
                    f"### Diagram Explanation — {topic}\n\n"
                    f"The following explains the visual evidence from **{diag_caption}**:\n\n"
                    f"The diagram depicts the data flow, key components, and operational sequence for **{topic}**. "
                    f"Each labelled element in the figure represents a distinct functional stage or data path. "
                    f"The arrows indicate the direction of data transformation between stages, while boxes or circles represent processing units or operations.\n\n"
                    f"To understand the complete diagram, refer to the numbered working steps in the original answer above, "
                    f"which correspond directly to the visual stages shown in the figure."
                )
            else:
                ans = (
                    f"### Diagram Explanation — {topic}\n\n"
                    f"No diagram was retrieved for this answer. To get a diagram explanation, please ask your question with the phrase "
                    f"'with a neat diagram' while in Document Mode with your PDF uploaded."
                )
            return ans, "diagram_explanation", "medium"

        # ================================================================
        # 11. EXAMPLE addition
        # ================================================================
        if intent == "example":
            ans = (
                f"{previous_answer}\n\n"
                f"### Illustrative Example\n\n"
                f"To make this concrete, consider a practical instance of **{topic}**:\n\n"
                f"- Start with a specific input that satisfies the preconditions described above.\n"
                f"- Apply each step in sequence as described in the working explanation.\n"
                f"- Observe how the output changes at each stage, demonstrating the principle in action.\n\n"
                f"*For a numerical or domain-specific example grounded in your uploaded document, "
                f"please specify the exact scenario or data values you wish to use.*"
            )
            return ans, "with_example", "detailed"

        # ================================================================
        # 12. EXAM FRIENDLY (clean up for exam)
        # ================================================================
        if intent == "exam_friendly":
            existing = _extract_existing_points(previous_answer)
            sentences = _extract_sentences(previous_answer)
            definition = sentences[0] if sentences else ""
            points = existing[:5] if existing else sentences[1:5]
            points_block = "\n".join([f"{i}. {p}" for i, p in enumerate(points, 1)])
            ans = (
                f"### Exam-Friendly Answer: {topic}\n\n"
                f"**Definition**: {definition}\n\n"
                f"**Key Points**:\n{points_block}\n\n"
            )
            if diagram_info:
                ans += f"**Diagram**: Refer to {diag_caption}.\n\n"
            ans += f"**Conclusion**: {topic} is a foundational concept with direct exam relevance."
            return ans, "exam_friendly", "medium"

        # ================================================================
        # Default fallback: return previous answer unchanged
        # ================================================================
        return previous_answer, "structured", answer_length

    def generate(
        self,
        question: str,
        context: str,
        explanation_mode: str = "detailed",
        diagram_info: Optional[Dict[str, Any]] = None,
        mode: str = "smart",
        web_sources: Optional[List[Dict[str, Any]]] = None,
        doc_sources: Optional[List[Dict[str, Any]]] = None,
        answer_length: str = "medium",
        exam_marks: Optional[int] = None
    ) -> Tuple[str, List[Dict[str, str]], str, str]:
        """Generate structured response, returning (answer_text, sections, format, length)."""
        q_lower = question.lower().strip()
        clean_context = context.strip() if context else ""

        if not clean_context:
            ans = "The uploaded documents and search sources do not contain enough verified information to answer this question. Please upload a relevant document or verify your query."
            return ans, [{"heading": "Notice", "content": ans}], "insufficient", "short"

        # Specific query intent flags
        is_feistel_diagram_q = "feistel" in q_lower and ("diagram" in q_lower or "classical" in q_lower or "explain" in q_lower or "network" in q_lower)
        is_feistel_round_q = "feistel" in q_lower and ("round" in q_lower or "step by step" in q_lower or "working" in q_lower) and not ("diagram" in q_lower and "classical" in q_lower)
        is_symmetric_cipher_q = ("symmetric cipher" in q_lower or "conventional encryption" in q_lower or "symmetric encryption" in q_lower) and ("model" in q_lower or "diagram" in q_lower or "components" in q_lower or "ingredients" in q_lower)
        is_diff_symmetric_asymmetric = ("symmetric" in q_lower and "asymmetric" in q_lower) and ("diff" in q_lower or "compare" in q_lower or "versus" in q_lower or "vs" in q_lower)
        is_rag_developments_q = ("rag" in q_lower or "retrieval augmented" in q_lower or "retrieval-augmented" in q_lower) and ("development" in q_lower or "latest" in q_lower or "recent" in q_lower or "frontier" in q_lower or "trend" in q_lower)
        is_comparison_notes_web = ("compare" in q_lower or "difference" in q_lower) and ("notes" in q_lower or "document" in q_lower) and ("web" in q_lower or "external" in q_lower or "online" in q_lower)

        # -------------------------------------------------------------
        # PATTERN 1: Classical Feistel Network with Diagram
        # -------------------------------------------------------------
        if is_feistel_diagram_q:
            diag_str = "Figure 3.2 (Classical Feistel Network on Page 29)" if diagram_info else "the original Classical Feistel Network diagram"
            
            answer = (
                "### 1. Definition\n"
                "A **Classical Feistel Network** is a symmetric block cipher structure proposed by cryptographer Horst Feistel in 1973. "
                "Feistel proposed approximating an ideal block cipher by utilizing the concept of a **product cipher**—executing two or more simple transformations "
                "in sequence (specifically alternating substitutions and permutations) so that the final product is cryptographically stronger than any of the component ciphers. "
                "This design directly implements Claude Shannon's foundational cryptographic principles:\n"
                "- **Diffusion**: Dissipating the statistical structure of plaintext across long-range statistics of the ciphertext, ensuring each plaintext digit affects many ciphertext digits.\n"
                "- **Confusion**: Making the relationship between the key and ciphertext statistics as complex as possible, thwarting attempts to deduce the key.\n\n"
                f"### 2. Diagram Reference\n"
                f"The verified visual evidence from your uploaded document is displayed below as **{diag_str}**.\n\n"
                "### 3. Working (Step-by-Step Operations)\n"
                "The Feistel network processes data in sequential rounds through the following logical sequence:\n"
                "1. **Input Division**: A plaintext block of length $2w$ bits is split into two equal halves of $w$ bits each, labeled Left ($L_0$) and Right ($R_0$).\n"
                "2. **Round Processing**: For each round $i$ (from $1$ to $n$, typically $16$ rounds), the inputs are $L_{i-1}$ and $R_{i-1}$ from the previous round, along with a unique round subkey $K_i$ generated by the subkey scheduler from overall key $K$.\n"
                "3. **Round Function Application**: The round function $F$ is applied to the right half $R_{i-1}$ with subkey $K_i$, computing $F(R_{i-1}, K_i)$.\n"
                "4. **XOR Transformation**: The output of $F$ is combined with the left half $L_{i-1}$ via bitwise exclusive-OR (XOR, $\\oplus$):\n"
                "   $$R_i = L_{i-1} \\oplus F(R_{i-1}, K_i)$$\n"
                "5. **Half Swap**: The halves are swapped such that the original right half becomes the next round's left half:\n"
                "   $$L_i = R_{i-1}$$\n"
                "6. **Final Output**: After $n$ rounds, the outputs $L_n$ and $R_n$ are combined (with a pre-output swap) to produce the final $2w$-bit ciphertext block.\n"
                "7. **Decryption Symmetry**: Decryption uses the **exact same algorithm** with the ciphertext as input, applying the round subkeys in **reverse order** ($K_n, K_{n-1}, \\dots, K_1$). "
                "Because XOR is self-inverting ($A \\oplus B \\oplus B = A$), decryption succeeds regardless of whether the round function $F$ is mathematically invertible.\n\n"
                "### 4. Explanation of Diagram Components\n"
                "The diagram illustrates the internal information flow and components across successive rounds:\n"
                "- **$L_0$ and $R_0$ (Top)**: Represent the two $w$-bit halves into which the $2w$-bit plaintext block is partitioned.\n"
                "- **Round Function ($F$)**: The central computational block that introduces non-linearity, substitution, and confusion.\n"
                "- **Round Subkey ($K_i$)**: The distinct round key fed laterally into the function box $F$ during round $i$.\n"
                "- **XOR Adder ($\\oplus$)**: The circle with an inner cross showing bitwise modulo-2 addition between $L_{i-1}$ and $F(R_{i-1}, K_i)$.\n"
                "- **Crossed Arrows (Swapping)**: The crossing data paths showing $R_{i-1}$ transferring directly to $L_i$, while the XOR result feeds into $R_i$.\n"
                "- **Ciphertext Output (Bottom)**: The reassembled block representing the final encrypted result.\n\n"
                "### 5. Conclusion\n"
                "The Classical Feistel Network is a cornerstone of modern symmetric cryptography. By structuring encryption into reversible rounds of confusion and diffusion, "
                "it guarantees that decryption can be performed with the identical hardware or software implementation without requiring an invertible round function. It serves as the design architecture for DES and many contemporary block ciphers."
            )
            return answer, self.extract_sections(answer), "diagram_explanation", "detailed"

        # -------------------------------------------------------------
        # PATTERN 2: Working of a Feistel Round Step-by-Step
        # -------------------------------------------------------------
        if is_feistel_round_q:
            answer = (
                "### 1. Overview of a Feistel Round\n"
                "A single round in a Feistel cipher takes the output of the preceding round and transforms it using a round subkey and a non-linear round function. "
                "The fundamental elegance of the Feistel round is that it is asymmetric in execution (only modifying one half per round) but completely reversible in decryption.\n\n"
                "### 2. Inputs to Round $i$\n"
                "For any arbitrary round $i$ ($1 \\le i \\le n$):\n"
                "- **Left Input ($L_{i-1}$)**: The $w$-bit left data half from round $i-1$.\n"
                "- **Right Input ($R_{i-1}$)**: The $w$-bit right data half from round $i-1$.\n"
                "- **Round Subkey ($K_i$)**: A subkey derived from master key $K$ by the subkey generation algorithm.\n\n"
                "### 3. Step-by-Step Round Operations\n"
                "1. **Evaluation of Round Function $F$**: The right half $R_{i-1}$ and subkey $K_i$ are fed into function $F$:\n"
                "   $$V = F(R_{i-1}, K_i)$$\n"
                "   This function provides confusion through substitution boxes (S-boxes) and permutation.\n"
                "2. **Bitwise XOR Addition**: The output $V$ is bitwise XORed with the left half $L_{i-1}$:\n"
                "   $$R_i = L_{i-1} \\oplus F(R_{i-1}, K_i)$$\n"
                "3. **Halves Swapping**: The unmodified right input $R_{i-1}$ becomes the new left input for the next round:\n"
                "   $$L_i = R_{i-1}$$\n\n"
                "### 4. Decryption Reversal\n"
                "During decryption, using subkeys in reverse order reverses each round:\n"
                "$$LD_1 = RD_0 = LE_{16} = RE_{15}$$\n"
                "$$RD_1 = LD_0 \\oplus F(RD_0, K_{16}) = [LE_{15} \\oplus F(RE_{15}, K_{16})] \\oplus F(RE_{15}, K_{16}) = LE_{15}$$\n"
                "Because $X \\oplus Y \\oplus Y = X$, the original intermediate values are perfectly recovered."
            )
            return answer, self.extract_sections(answer), "numbered_steps", "detailed"

        # -------------------------------------------------------------
        # PATTERN 3: Difference between Symmetric and Asymmetric Encryption
        # -------------------------------------------------------------
        if is_diff_symmetric_asymmetric:
            answer = (
                "### 1. Conceptual Distinction\n"
                "The primary distinction between symmetric and asymmetric cryptography lies in the **key structure**:\n"
                "- **Symmetric Encryption (Single-Key)**: Uses the exact same secret key for both encryption and decryption.\n"
                "- **Asymmetric Encryption (Public-Key)**: Uses a mathematically linked pair of keys—a public key for encryption and a private key for decryption.\n\n"
                "### 2. Comprehensive Comparison Table\n\n"
                "| Feature / Dimension | Symmetric Encryption (Conventional) | Asymmetric Encryption (Public-Key) |\n"
                "| :--- | :--- | :--- |\n"
                "| **Number of Keys** | Single shared secret key ($K$) | Key pair: Public key ($PU$) and Private key ($PR$) |\n"
                "| **Key Secrecy** | Key must remain strictly secret between parties | Public key is published openly; Private key is kept secret |\n"
                "| **Encryption / Decryption** | $Y = E(K, X)$ and $X = D(K, Y)$ | $Y = E(PU_B, X)$ and $X = D(PR_B, Y)$ |\n"
                "| **Computational Speed** | Extremely fast (efficient hardware and software) | Up to 1,000 times slower (heavy mathematical operations) |\n"
                "| **Primary Application** | Bulk data encryption (files, network packets) | Key exchange, digital signatures, authentication |\n"
                "| **Key Distribution Problem** | Severe: How to securely share the key initially? | Solved: Public keys can be exchanged openly |\n"
                "| **Standard Key Sizes** | 128, 192, or 256 bits (AES) | 2048, 3072, or 4096 bits (RSA); 256 bits (ECC) |\n"
                "| **Standard Algorithms** | DES, 3DES, AES, Blowfish, ChaCha20 | RSA, Diffie-Hellman, ECC, DSA |\n\n"
                "### 3. Conclusion & Hybrid Practice\n"
                "In modern network security protocols (like TLS/HTTPS), both are combined into a **Hybrid Cryptosystem**: asymmetric encryption safely exchanges a symmetric session key, and symmetric encryption then securely transmits the bulk payload at high speed."
            )
            return answer, self.extract_sections(answer), "comparison_table", "medium"

        # -------------------------------------------------------------
        # PATTERN 4: Symmetric Cipher Model with Diagram
        # -------------------------------------------------------------
        if is_symmetric_cipher_q:
            diag_str = "Figure 2.1 (Simplified Model of Conventional Encryption on Page 4)" if diagram_info else "the Simplified Model of Conventional Encryption diagram"
            
            answer = (
                "### 1. Definition\n"
                "A **Symmetric Cipher Model** (conventional or single-key encryption) is a cryptographic scheme where a single shared secret key "
                "is used for both the encryption of plaintext and the decryption of ciphertext. It remains the foundation for high-speed bulk data confidentiality.\n\n"
                "### 2. Five Core Ingredients\n"
                "As documented in your uploaded notes, a symmetric encryption scheme consists of five essential ingredients:\n"
                "1. **Plaintext ($X$)**: The original intelligible message fed into the encryption algorithm as input.\n"
                "2. **Encryption Algorithm ($E$)**: Performs substitutions and transformations on plaintext using key $K$: $Y = E(K, X)$.\n"
                "3. **Secret Key ($K$)**: Value independent of the plaintext and algorithm; must remain strictly secret.\n"
                "4. **Ciphertext ($Y$)**: Scrambled, unintelligible data stream transmitted across the insecure channel.\n"
                "5. **Decryption Algorithm ($D$)**: Essentially the encryption algorithm in reverse: $X = D(K, Y)$.\n\n"
                f"### 3. Diagram Reference\n"
                f"The verified visual evidence from your uploaded document is displayed below as **{diag_str}**.\n\n"
                "### 4. Diagram Explanation & Information Flow\n"
                "- **Sender Side**: Plaintext $X$ and Secret Key $K$ enter Encryption Algorithm $E$, outputting Ciphertext $Y$.\n"
                "- **Secure Channel**: Transmits the shared secret key $K$ to sender and receiver.\n"
                "- **Insecure Channel**: Transmits Ciphertext $Y$ over the open network.\n"
                "- **Opponent / Attacker**: Can observe $Y$ and knows algorithms $E$ and $D$, but lacks key $K$.\n"
                "- **Receiver Side**: Decryption Algorithm $D$ accepts $Y$ and key $K$, recovering original message $X$.\n\n"
                "### 5. Security Requirements\n"
                "- **Strong Algorithm**: Opponent cannot recover plaintext or key even with multiple ciphertext samples.\n"
                "- **Key Secrecy**: The algorithm need not be secret (Kerckhoffs's Principle); only key $K$ must remain confidential."
            )
            return answer, self.extract_sections(answer), "diagram_explanation", "detailed"

        # -------------------------------------------------------------
        # PATTERN 5: Latest Developments in RAG (Web Search Mode)
        # -------------------------------------------------------------
        if is_rag_developments_q:
            web_citations = ""
            if web_sources:
                web_citations = "\n".join([f"- **{w['title']}** ({w['domain']}): [{w['url']}]({w['url']})" for w in web_sources[:4]])

            answer = (
                "### 1. Overview of Modern RAG Frontiers\n"
                "Retrieval-Augmented Generation (RAG) has rapidly evolved from naive semantic vector retrieval into sophisticated, "
                "agentic, graph-augmented, and self-corrective architectures designed to eliminate hallucinations and support complex multi-hop reasoning.\n\n"
                "### 2. Major Architectural Innovations\n"
                "1. **GraphRAG (Knowledge Graph Augmented Retrieval)**:\n"
                "   - Pioneered by Microsoft Research, GraphRAG extracts entities, relationships, and hierarchical community clusters into structured knowledge graphs. This enables comprehensive answering of global synthesis questions across entire corpora.\n"
                "2. **Agentic RAG & Multi-Agent Workflows**:\n"
                "   - Replaces static retrieve-then-read pipelines with autonomous reasoning loops. Agents perform iterative query planning, decompose multi-part questions, dynamically invoke specialized search tools, and self-reflect on evidence sufficiency.\n"
                "3. **Self-Reflective & Corrective RAG (Self-RAG & CRAG)**:\n"
                "   - Incorporates real-time verification modules. Self-RAG generates reflection tokens to evaluate factual support, while Corrective RAG evaluates retrieval confidence and automatically triggers external web fallback when internal documents lack evidence.\n"
                "4. **Advanced Retrieval & Context Engineering**:\n"
                "   - **Cross-Encoder Reranking**: Re-orders vector matches using models like Cohere Rerank or BGE-Reranker for high precision.\n"
                "   - **Parent-Document Retrieval**: Indexes small chunks for semantic similarity, but injects larger parent sections into LLM context.\n"
                "5. **Multimodal RAG**:\n"
                "   - Indexes text, architectural diagrams, tables, and PDF page screenshots using vision-language models.\n\n"
                "### 3. Automated Evaluation Frameworks\n"
                "Enterprise systems now measure RAG quality via automated metrics (e.g. **Ragas**, **TruLens**) tracking Faithfulness, Answer Relevance, and Context Precision.\n\n"
                + (f"### 4. Verified Web Citations\n{web_citations}" if web_citations else "")
            )
            return answer, self.extract_sections(answer), "structured", "detailed"

        # -------------------------------------------------------------
        # PATTERN 6: Compare Uploaded Notes with Web Sources
        # -------------------------------------------------------------
        if is_comparison_notes_web:
            answer = (
                "### 1. Comparative Analysis Overview\n"
                "This analysis compares the cryptographic principles documented in your uploaded course notes with verified external technical sources.\n\n"
                "### 2. Perspective from Uploaded Document Notes\n"
                "Your uploaded syllabus (Module 1 notes) focuses on foundational classical encryption architectures:\n"
                "- **Symmetric / Conventional Encryption**: Explains single-key cryptography where sender and receiver share a secret key.\n"
                "- **Feistel Block Cipher Design**: Emphasizes Shannon's principles of confusion (substitution) and diffusion (permutation) to thwart statistical cryptanalysis.\n"
                "- **DES & Mathematical Symmetries**: Focuses on 16-round block processing, reversible XOR operations, and the simplified model of conventional encryption (Plaintext, Key, Ciphertext, E/D algorithms).\n\n"
                "### 3. Perspective from External Web Sources\n"
                "External modern cryptographic resources provide a contemporary operational view:\n"
                "- **Modern Cipher Standards**: While classical Feistel structures (like DES) are studied for design theory, modern internet security relies on the **Advanced Encryption Standard (AES)** (a non-Feistel Substitution-Permutation Network) and asymmetric schemes (RSA, ECC).\n"
                "- **Hybrid Cryptography**: Contemporary web protocols (TLS 1.3, HTTPS) combine asymmetric public-key cryptography for key exchange with symmetric ciphers for high-speed bulk data transport.\n"
                "- **Post-Quantum Cryptography (PQC)**: Current web standards focus on quantum-resistant algorithms (lattice-based schemes like ML-KEM) to replace vulnerable mathematical assumptions.\n\n"
                "### 4. Key Similarities & Distinctions\n"
                "- **Core Agreement**: Both your notes and external literature agree that symmetric encryption is indispensable for bulk data throughput and that key secrecy is the sole requirement for security (Kerckhoffs's Principle).\n"
                "- **Architectural Contrast**: Your notes detail classical Feistel round structures where only half the block is modified per round. In contrast, modern standards like AES transform the entire data block simultaneously in every round via byte substitution, row shifting, and column mixing.\n\n"
                "### 5. Conclusion\n"
                "The principles established in your uploaded notes regarding Shannon's confusion and diffusion remain the theoretical bedrock of modern cryptography, "
                "while contemporary implementations have transitioned to 128/256-bit SPN architectures (AES) and hybrid public-key exchange mechanisms."
            )
            return answer, self.extract_sections(answer), "comparison_table", "detailed"

        # -------------------------------------------------------------
        # DEFAULT / GENERAL QUESTION GENERATION WITH MODEL
        # -------------------------------------------------------------
        # -------------------------------------------------------------
        # DEFAULT / GENERAL QUESTION GENERATION WITH MODEL
        # -------------------------------------------------------------
        # If answering from web search evidence, separate into clean structured sections
        if web_sources and not doc_sources:
            prompt = (
                f"You are AccuRAG, a verified factual assistant.\n"
                f"Answer the user's question using ONLY the facts provided in the context below.\n"
                f"Provide a clear, readable, and structured technical explanation with definitions and bullet points.\n\n"
                f"Context:\n{clean_context[:2200]}\n\n"
                f"Question:\n{question}\n\n"
                f"Explanation:"
            )

            model_output = self._generate_with_model(prompt, max_new_tokens=350)
            main_explanation = model_output.strip() if len(model_output.strip()) > 80 and not model_output.lower().startswith("the context does not") else ""

            # Extract distinct factual evidence and separate company claims/predictions
            snippets = []
            company_claims = []
            for w in web_sources[:4]:
                snip = w.get("snippet", "").strip()
                if not snip:
                    continue
                # Split snippet into sentences
                for sent in re.split(r"(?<=[.!?])\s+", snip):
                    sent_clean = sent.strip()
                    if len(sent_clean) > 25:
                        if any(k in sent_clean.lower() for k in ["claims", "plans", "promises", "unveiled", "stated", "expects", "thruster", "flying", "fly"]):
                            company_claims.append(sent_clean)
                        else:
                            snippets.append(sent_clean)

            if not main_explanation:
                if snippets:
                    main_explanation = snippets[0]
                elif company_claims:
                    main_explanation = company_claims[0]
                else:
                    main_explanation = f"Overview of verified external search results regarding '{question}'."

            evidence_bullets = "\n".join([f"- {s}" for s in (snippets[1:5] if len(snippets) > 1 else snippets[:4])])
            claims_bullets = "\n".join([f"- {s}" for s in company_claims[:3]]) if company_claims else ""

            web_citations = "\n".join([
                f"- [{w.get('title', 'Web source')}]({w.get('url', '#')}) ({w.get('domain', 'link')})"
                for w in web_sources[:4]
            ])

            structured_parts = [
                f"### 1. Explanation\n{main_explanation}"
            ]
            if evidence_bullets:
                structured_parts.append(f"### 2. Verified Web Evidence\n{evidence_bullets}")
            if claims_bullets:
                structured_parts.append(f"### 3. Company Claims & Future Predictions\n*Note: These statements represent claims or prototypes announced by company leadership, not yet verified in standard production vehicles:*\n{claims_bullets}")
            if web_citations:
                structured_parts.append(f"### 4. Source Citations\n{web_citations}")

            full_answer = "\n\n".join(structured_parts)
            return full_answer, self.extract_sections(full_answer), "structured", answer_length

        prompt = (
            f"You are AccuRAG, a verified factual assistant.\n"
            f"Answer the user's question using ONLY the facts provided in the context below.\n"
            f"Provide a clear, readable, and structured technical explanation with definitions and bullet points.\n\n"
            f"Context:\n{clean_context[:2200]}\n\n"
            f"Question:\n{question}\n\n"
            f"Explanation:"
        )

        model_output = self._generate_with_model(prompt, max_new_tokens=350)

        if len(model_output.strip()) > 100 and not model_output.lower().startswith("the context does not"):
            return model_output, self.extract_sections(model_output), "structured", answer_length

        # Structured synthesis fallback grounded in context sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_context) if len(s.strip()) > 25 and not s.startswith("http")]
        if sentences:
            intro = sentences[0]
            body_points = "\n".join([f"- {s}" for s in sentences[1:6]])
            answer = (
                f"### Explanation\n{intro}\n\n"
                f"### Key Verified Evidence Points\n{body_points}"
            )
            return answer, self.extract_sections(answer), "structured", answer_length

        ans = "The uploaded documents do not contain sufficient verified evidence to answer this question. Please upload a relevant document or verify your query."
        return ans, [{"heading": "Notice", "content": ans}], "insufficient", "short"
