import re
from typing import List, Dict, Any, Optional

class DecompositionEngine:
    def __init__(self):
        pass

    def is_complex_query(self, query: str) -> bool:
        """Determine whether a question contains multiple distinct sub-tasks requiring decomposition."""
        q = query.strip()

        # Multiple question marks
        if q.count("?") > 1:
            return True

        # Check for multiple action clauses: e.g. "explain X and describe Y", "what is X and how does Y work"
        action_verbs = [
            r"\bexplain\b", r"\bdescribe\b", r"\bcompare\b", r"\blist\b",
            r"\bhow does\b", r"\bwhat is\b", r"\bdifferentiate\b", r"\banalyze\b",
            r"\boutline\b", r"\bdetail\b"
        ]

        # Match multiple action verbs in the query
        verb_matches = 0
        for verb_pattern in action_verbs:
            matches = re.findall(verb_pattern, q, re.IGNORECASE)
            verb_matches += len(matches)

        # If query has 2+ action verbs and coordinating conjunctions or commas, it's a compound question
        has_conjunction = any(c in q.lower() for c in [" and ", " as well as ", ", and ", "; ", " also "])
        if verb_matches >= 2 and has_conjunction and len(q.split()) >= 8:
            return True

        # Commas separating imperative requests
        if re.search(r",\s*(and\s+)?(describe|compare|explain|list|show|detail)\b", q, re.IGNORECASE):
            return True

        return False

    def decompose(self, query: str) -> List[str]:
        """Break down a compound question into atomic sub-questions."""
        q = query.strip()
        if not self.is_complex_query(q):
            return [q]

        sub_queries = []

        # Split on comma conjunctions, semicolons, or "and [verb]"
        parts = re.split(r";\s*|\.\s+|\s+and\s+(?=(?:describe|explain|compare|list|what|how|show)\b)|,\s*(?:and\s+)?(?=(?:describe|explain|compare|list|what|how|show)\b)", q, flags=re.IGNORECASE)

        main_topic_match = re.search(r"(?:about|of|on|for)\s+([A-Za-z0-9\s-]+?)(?:,|\.|\sand\s|$)", q, re.IGNORECASE)
        main_topic = main_topic_match.group(1).strip() if main_topic_match else ""

        for part in parts:
            part = part.strip()
            # Clean trailing punctuation
            part = re.sub(r"^[,\s]+|[,\s]+$", "", part)
            if len(part.split()) < 3:
                continue

            # Ensure sub-question is self-contained by contextualizing with main topic if needed
            if main_topic and not any(term in part.lower() for term in main_topic.lower().split()):
                if not re.search(r"\b(it|its|this|these|the model|the algorithm)\b", part, re.IGNORECASE):
                    part = f"{part} regarding {main_topic}"

            if not part.endswith("?"):
                part += "?"
            sub_queries.append(part)

        if len(sub_queries) <= 1:
            return [q]

        return sub_queries[:4]  # Maximum 4 sub-queries for focused RAG
