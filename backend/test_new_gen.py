import os
import sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

from document_engine import DocumentEngine

upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
diagram_dir = os.path.join(os.path.dirname(__file__), "diagrams")
doc_engine = DocumentEngine(upload_dir=upload_dir, diagram_dir=diagram_dir)

pdf_path = os.path.join(upload_dir, "BCS703_CNS_Module1.pdf")
with open(pdf_path, "rb") as f:
    doc_engine.process_pdf("BCS703_CNS_Module1.pdf", f.read(), replace_existing=True)

q = "Explain Classical Feistel Network with a neat diagram."
matches = doc_engine.retrieve(q, top_k=4)
diag_page = doc_engine.locate_diagram_page(q, matches[0]["page"], "BCS703_CNS_Module1.pdf")
print("Diagram page located:", diag_page)

surrounding = doc_engine.get_surrounding_page_chunks("BCS703_CNS_Module1.pdf", diag_page, span=1)
print(f"Surrounding chunks retrieved: {len(surrounding)} from pages {[c['page'] for c in surrounding]}")

# Combine unique matches
all_evidence = []
seen = set()
for c in matches + surrounding:
    txt = c["text"]
    if txt not in seen:
        seen.add(txt)
        all_evidence.append(c)

print(f"Total unified evidence passages: {len(all_evidence)}")
for i, ev in enumerate(all_evidence):
    print(f"[{i+1}] P{ev['page']}: {ev['text'][:120]}...")
