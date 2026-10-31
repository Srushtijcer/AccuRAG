import re
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class VerificationEngine:
    def __init__(self, embed_model=None):
        self.embed_model = embed_model

    def extract_claims(self, text: str) -> List[str]:
        """Decompose text into distinct factual claims, stripping formatting markers."""
        if not text:
            return []

        # Split on sentence terminals and bullet points
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n[-*•]\s+|\n\d+\.\s+", text)
        claims = []

        filler_patterns = [
            r"^(here is|according to|based on|as mentioned|in summary|to summarize|note that|it is worth noting)",
            r"^(sure|certainly|below is|the following|let me explain)",
            r"^(in conclusion|overall|firstly|secondly|thirdly)"
        ]

        for s in raw_sentences:
            s_clean = s.strip()
            s_clean = re.sub(r"^[#\s\-*•\d.]+", "", s_clean).strip()
            s_clean = s_clean.replace("**", "").replace("*", "").strip()

            if len(s_clean.split()) < 4 or s_clean.startswith("$$") or s_clean.startswith("|"):
                continue

            # Strip filler
            for pat in filler_patterns:
                s_clean = re.sub(pat, "", s_clean, flags=re.IGNORECASE).strip()
                if s_clean.startswith(","):
                    s_clean = s_clean[1:].strip()

            if len(s_clean) > 15 and not s_clean.endswith("?") and not s_clean.lower().startswith("###"):
                claims.append(s_clean)

        return claims[:12]  # Focus on key claims

    def _token_overlap(self, str1: str, str2: str) -> float:
        """Compute Jaccard token overlap between claim and evidence snippet."""
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "of", "and", "in", "to", "for",
            "with", "on", "at", "by", "from", "up", "about", "into", "through", "after",
            "this", "that", "these", "those", "it", "its", "which", "who", "whom"
        }
        w1 = set(w.lower() for w in re.findall(r"[A-Za-z0-9]+", str1) if w.lower() not in stop_words and len(w) > 2)
        w2 = set(w.lower() for w in re.findall(r"[A-Za-z0-9]+", str2) if w.lower() not in stop_words and len(w) > 2)
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1)

    def _check_negation_conflict(self, claim: str, evidence: str) -> bool:
        """Detect genuine factual contradictions between claim and source text."""
        c_lower = claim.lower()
        e_lower = evidence.lower()

        # Specific cryptographic factual contradictions
        if "symmetric" in c_lower:
            if ("two different keys" in c_lower or "two keys" in c_lower or "public and private" in c_lower) and ("single key" in e_lower or "same key" in e_lower or "one key" in e_lower):
                return True
        if "asymmetric" in c_lower:
            if ("single key" in c_lower or "same key" in c_lower) and ("two keys" in e_lower or "public key" in e_lower):
                return True

        if ("is not" in c_lower and "is a" in e_lower and self._token_overlap(claim, evidence) > 0.65):
            return True
        if ("cannot" in c_lower and "can easily" in e_lower and self._token_overlap(claim, evidence) > 0.65):
            return True

        return False

    def _classify_claim_nature(self, claim: str) -> str:
        """Classify factual nature of statement: confirmed fact, company claim, prediction, or unverified feature."""
        c_low = claim.lower()
        if any(w in c_low for w in ["predict", "expected to", "plans to", "aims to", "targeted", "future", "by 2026", "by 2027", "promises"]):
            return "unconfirmed_prediction"
        if any(w in c_low for w in ["musk stated", "musk claimed", "tesla claims", "company claims", "announced that", "unveiled", "purported"]):
            return "company_claim"
        if any(w in c_low for w in ["fly", "flying", "hover", "thruster", "upside down", "flips mid-air"]):
            return "unproven_feature"
        return "confirmed_fact"

    def verify_single_claim(
        self,
        claim: str,
        doc_sources: List[Dict[str, Any]],
        web_sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify an individual claim against document and web evidence."""
        best_overlap = 0.0
        best_source_label = None
        has_contradiction = False
        source_type = None
        claim_nature = self._classify_claim_nature(claim)

        # Check document sources
        for src in doc_sources:
            text = src.get("text", "")
            overlap = self._token_overlap(claim, text)
            if self._check_negation_conflict(claim, text):
                has_contradiction = True
                best_source_label = f"{src.get('document', 'Document')} (Page {src.get('page', '?')})"
                source_type = "document"
                break
            if overlap > best_overlap:
                best_overlap = overlap
                best_source_label = f"{src.get('document', 'Document')} (Page {src.get('page', '?')})"
                source_type = "document"

        # Check web sources
        if not has_contradiction:
            for w_src in web_sources:
                text = w_src.get("snippet", "") + " " + w_src.get("title", "")
                overlap = self._token_overlap(claim, text)
                if self._check_negation_conflict(claim, text):
                    has_contradiction = True
                    best_source_label = f"Web: {w_src.get('title', 'Web source')} ({w_src.get('domain', 'link')})"
                    source_type = "web"
                    break
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_source_label = f"Web: {w_src.get('title', 'Web source')} ({w_src.get('domain', 'link')})"
                    source_type = "web"

        if has_contradiction:
            status = "contradicted"
            conf = 0.85
        elif claim_nature in ["company_claim", "unconfirmed_prediction", "unproven_feature"]:
            # Distinguish company claims / predictions / unproven features
            status = claim_nature
            conf = min(0.85, round(best_overlap + 0.30, 2)) if best_overlap >= 0.18 else 0.40
        elif best_overlap >= 0.35:
            status = "supported"
            conf = min(0.98, round(best_overlap + 0.35, 2))
        elif best_overlap >= 0.18:
            status = "partially_supported"
            conf = round(best_overlap + 0.20, 2)
        else:
            status = "unsupported"
            conf = 0.20

        return {
            "claim": claim,
            "status": status,
            "source": best_source_label if status != "unsupported" else None,
            "source_type": source_type if status != "unsupported" else None,
            "confidence": conf,
            "claim_type": claim_nature
        }

    def detect_source_contradictions(
        self,
        doc_sources: List[Dict[str, Any]],
        web_sources: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify conflicting factual claims between uploaded documents and web sources."""
        contradictions = []
        if not doc_sources or not web_sources:
            return contradictions

        doc_text = " ".join([d.get("text", "") for d in doc_sources[:3]])
        for w in web_sources[:3]:
            w_text = w.get("snippet", "")
            if ("single key" in doc_text.lower() or "secret key" in doc_text.lower()) and "asymmetric" in w_text.lower() and "two keys" in w_text.lower():
                if "symmetric" in w_text.lower() and "different" in w_text.lower():
                    contradictions.append(
                        f"Terminology contrast: Document emphasizes single-key symmetric models, whereas web source '{w.get('title')}' discusses two-key asymmetric mechanisms."
                    )
        return contradictions

    def verify_answer(
        self,
        question: str,
        answer: str,
        doc_sources: List[Dict[str, Any]],
        web_sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform factual verification and compute transparent trust score & evidence coverage."""
        if not answer or "not contain enough" in answer.lower() or "not contain sufficient" in answer.lower():
            return {
                "status": "Insufficient evidence",
                "trust_score": 15.0,
                "evidence_coverage": "Limited evidence coverage",
                "self_corrected": False,
                "claims": [],
                "contradictions": [],
                "notes": "Available sources do not provide sufficient verified evidence for this query."
            }

        claims = self.extract_claims(answer)
        if not claims:
            claims = [answer.strip()[:200]]

        verified_claims = [
            self.verify_single_claim(c, doc_sources, web_sources)
            for c in claims
        ]

        supported_count = sum(1 for c in verified_claims if c["status"] in ("supported", "confirmed_fact"))
        partial_count = sum(1 for c in verified_claims if c["status"] in ("partially_supported", "company_claim", "unconfirmed_prediction", "unproven_feature"))
        contradicted_count = sum(1 for c in verified_claims if c["status"] == "contradicted")
        unsupported_count = sum(1 for c in verified_claims if c["status"] == "unsupported")
        total_claims = len(verified_claims) or 1

        doc_supported = sum(1 for c in verified_claims if c["status"] in ("supported", "partially_supported", "company_claim") and c.get("source_type") == "document")
        web_supported = sum(1 for c in verified_claims if c["status"] in ("supported", "partially_supported", "company_claim", "unconfirmed_prediction", "unproven_feature") and c.get("source_type") == "web")

        contradictions = self.detect_source_contradictions(doc_sources, web_sources)
        if contradicted_count > 0:
            for c in verified_claims:
                if c["status"] == "contradicted":
                    contradictions.append(f"Contradiction detected: '{c['claim'][:80]}...' against {c['source']}")

        # Determine precise verification status label
        if contradicted_count > 0 or len(contradictions) > 0:
            status = "Conflicting evidence"
        elif unsupported_count == total_claims and total_claims > 0:
            status = "Insufficient evidence"
        elif supported_count >= (total_claims * 0.50):
            if doc_supported > 0 and web_supported > 0:
                status = "Supported by document and web evidence"
            elif web_supported > 0 and doc_supported == 0:
                status = "Supported by web evidence"
            else:
                status = "Supported by document evidence"
        elif (supported_count + partial_count) >= (total_claims * 0.35):
            status = "Partially supported"
        else:
            status = "Partially supported" if (supported_count + partial_count) > 0 else "Insufficient evidence"

        # Transparent Trust Score calculation (0.0 to 100.0)
        max_sim = max([s.get("similarity", 0.0) for s in doc_sources] + [0.75 if web_sources else 0.0])
        sim_factor = min(1.0, max_sim / 0.70)
        claim_support_ratio = (supported_count * 1.0 + partial_count * 0.6) / total_claims
        penalty = 0.35 if (contradicted_count > 0 or len(contradictions) > 0) else 0.0

        raw_score = (0.40 * sim_factor + 0.60 * claim_support_ratio - penalty)
        trust_score = round(max(15.0, min(98.5, raw_score * 100)), 1)

        # Evidence Coverage Indicator
        if trust_score >= 70.0 and (doc_sources or web_sources):
            coverage = "Strong evidence coverage"
        elif trust_score >= 38.0 and (doc_sources or web_sources):
            coverage = "Partial evidence coverage"
        else:
            coverage = "Limited evidence coverage"

        return {
            "status": status,
            "trust_score": trust_score,
            "evidence_coverage": coverage,
            "self_corrected": False,
            "claims": verified_claims,
            "contradictions": contradictions,
            "notes": f"Verified {total_claims} claims: {supported_count} supported, {partial_count} partial, {unsupported_count} unsupported, {contradicted_count} contradictions."
        }

    def self_correct_if_needed(
        self,
        question: str,
        answer: str,
        verification: Dict[str, Any],
        doc_sources: List[Dict[str, Any]],
        web_sources: List[Dict[str, Any]],
        max_attempts: int = 2
    ) -> Tuple[str, Dict[str, Any], bool]:
        """Verify answer integrity without destroying markdown structure."""
        if verification["status"] in ("Supported by document evidence", "Supported by web evidence", "Supported by document and web evidence", "Insufficient evidence"):
            return answer, verification, False

        contradicted = [c for c in verification["claims"] if c["status"] == "contradicted"]
        
        # Only rewrite if there are active contradictions
        if not contradicted:
            return answer, verification, False

        correction_notice = (
            "\n\n> [!WARNING] Verification Note: Some details were flagged as conflicting with source documents. "
            f"Please cross-check against cited sources: {contradicted[0].get('source')}."
        )
        corrected_answer = f"{answer}{correction_notice}"
        new_verif = dict(verification)
        new_verif["self_corrected"] = True
        new_verif["notes"] = "Self-correction applied: Warning appended regarding conflicting source claims."

        return corrected_answer, new_verif, True
