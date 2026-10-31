import os
import sys
import json
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

from main import doc_engine, gen_engine, verification_engine, decomp_engine

q = "Explain Classical Feistel Network with a neat diagram."
print("=== QUESTION ===")
print(q)

matches = doc_engine.retrieve(q, top_k=5)
print("\n=== RETRIEVED CHUNKS ===")
for i, m in enumerate(matches):
    print(f"\n[{i+1}] Doc: {m['document']} | Page: {m['page']} | Sim: {m['similarity']} | Hybrid: {m.get('hybrid_score')}")
    print(m['text'])

context_blocks = []
for idx, d in enumerate(matches, start=1):
    context_blocks.append(f"[{idx}] Source: {d['document']} (Page {d['page']}):\n{d['text']}")
context_str = "\n\n".join(context_blocks)

print("\n=== CONTEXT STR (FIRST 500 CHARS) ===")
print(context_str[:500])

raw_answer = gen_engine.generate(q, context_str, "detailed")
print("\n=== RAW GENERATED ANSWER ===")
print(raw_answer)

verif = verification_engine.verify_answer(q, raw_answer, matches, [])
print("\n=== INITIAL VERIFICATION ===")
print(json.dumps(verif, indent=2))

final_ans, final_verif, was_corr = verification_engine.self_correct_if_needed(q, raw_answer, verif, matches, [])
print("\n=== FINAL ANSWER AFTER SELF-CORRECT ===")
print("Was corrected:", was_corr)
print(final_ans)
