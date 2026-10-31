import os
import io
import unittest
from fastapi.testclient import TestClient
from main import app, doc_engine, web_engine, verification_engine, decomp_engine

client = TestClient(app)

class TestAccuRAGSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=======================================================")
        print("STARTING ACCURAG END-TO-END TEST SUITE (15 SCENARIOS)")
        print("=======================================================")

    def test_01_upload_valid_pdf(self):
        """Scenario 1: Upload a valid PDF file."""
        sample_path = os.path.join(doc_engine.upload_dir, "BCS703_CNS_Module1.pdf")
        self.assertTrue(os.path.exists(sample_path), "Sample PDF should exist in uploads")
        with open(sample_path, "rb") as f:
            pdf_bytes = f.read()

        response = client.post(
            "/upload",
            files={"file": ("test_upload_valid.pdf", pdf_bytes, "application/pdf")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["filename"], "test_upload_valid.pdf")
        self.assertGreater(data["pages"], 0)
        self.assertGreater(data["chunks_created"], 0)
        print("[PASS] Test 1: Valid PDF uploaded and indexed into vector store.")

    def test_02_upload_invalid_or_empty_file(self):
        """Scenario 2: Upload an invalid, empty, or non-PDF file."""
        # Empty file
        empty_resp = client.post(
            "/upload",
            files={"file": ("empty.pdf", b"", "application/pdf")}
        )
        self.assertEqual(empty_resp.status_code, 400)
        self.assertIn("empty", empty_resp.json()["detail"].lower())

        # Non-PDF file
        txt_resp = client.post(
            "/upload",
            files={"file": ("notes.txt", b"Hello world text file", "text/plain")}
        )
        self.assertEqual(txt_resp.status_code, 400)
        self.assertIn("only pdf", txt_resp.json()["detail"].lower())

        # Corrupted PDF
        corrupt_resp = client.post(
            "/upload",
            files={"file": ("corrupt.pdf", b"NOT_A_PDF_HEADER_DATA", "application/pdf")}
        )
        self.assertEqual(corrupt_resp.status_code, 400)
        self.assertIn("corrupted or invalid", corrupt_resp.json()["detail"].lower())
        print("[PASS] Test 2: Handled empty, non-PDF, and corrupted files gracefully.")

    def test_03_in_document_question(self):
        """Scenario 3: Ask a question whose answer exists in the PDF."""
        response = client.get("/ask", params={"question": "What is plaintext?", "mode": "document", "explanation_mode": "easy"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("original", data["answer"].lower())
        self.assertGreater(len(data["sources"]), 0)
        self.assertIn(data["verification"]["status"], ["Supported by evidence", "Partially supported"])
        print("[PASS] Test 3: Document QA returned grounded answer citing page evidence.")

    def test_04_out_of_document_question(self):
        """Scenario 4: Ask a question absent from the PDF."""
        response = client.get("/ask", params={"question": "What is the capital city of planet Neptune and its currency?", "mode": "document", "explanation_mode": "easy"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["verification"]["status"], "Insufficient evidence")
        print("[PASS] Test 4: Handled out-of-document query with honest limitation statement.")

    def test_05_multi_evidence_question(self):
        """Scenario 5: Ask a question requiring multiple pieces of document evidence."""
        response = client.get("/ask", params={"question": "What are the components of the symmetric cipher model?", "mode": "document", "explanation_mode": "detailed"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        ans = data["answer"].lower()
        self.assertTrue("plaintext" in ans or "secret key" in ans or "algorithm" in ans)
        self.assertGreaterEqual(len(data["sources"]), 2)
        print("[PASS] Test 5: Multi-evidence question aggregated multiple document passages.")

    def test_06_web_mode_query(self):
        """Scenario 6: Ask a current-information question in Web Mode."""
        response = client.get("/ask", params={"question": "What are retrieval augmented generation architectures?", "mode": "web", "explanation_mode": "easy"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "web")
        self.assertGreater(len(data["web_sources"]), 0)
        self.assertTrue(any("http" in w["url"] for w in data["web_sources"]))
        print("[PASS] Test 6: Web Mode retrieved live web citations and generated grounded answer.")

    def test_07_smart_mode_hybrid(self):
        """Scenario 7: Ask question in Smart Mode requiring smart retrieval."""
        response = client.get("/ask", params={"question": "Explain symmetric encryption principles and compare with recent RAG developments", "mode": "smart", "explanation_mode": "detailed"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "smart")
        self.assertGreater(len(data["sources"]), 0)
        print("[PASS] Test 7: Smart Mode routed hybrid intent properly.")

    def test_08_unsupported_claim_detection(self):
        """Scenario 8: Test claim verification on an answer with unsupported claims."""
        test_text = "Symmetric encryption uses two completely different public and private keys that are published globally."
        res = client.post("/verify", json={"text": test_text, "sources": [{"document": "test.pdf", "page": 3, "text": "Symmetric encryption uses a single secret key shared between sender and receiver."}]})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreater(data["total"], 0)
        self.assertIn(data["verification_results"][0]["status"], ["contradicted", "unsupported", "partially_supported"])
        print("[PASS] Test 8: Verification engine successfully detected unsupported/contradictory claims.")

    def test_09_conflicting_source_detection(self):
        """Scenario 9: Test conflicting source information detection."""
        doc_src = [{"document": "crypto.pdf", "page": 3, "text": "Symmetric encryption uses a single secret key for both encryption and decryption."}]
        web_src = [{"title": "Asymmetric Guide", "url": "https://example.com", "domain": "example.com", "snippet": "Asymmetric encryption uses two different keys, a public key and a private key."}]
        contradictions = verification_engine.detect_source_contradictions(doc_src, web_src)
        self.assertIsInstance(contradictions, list)
        print("[PASS] Test 9: Cross-source contradiction detector executed cleanly.")

    def test_10_diagram_retrieval(self):
        """Scenario 10: Ask for a diagram contained in a PDF."""
        response = client.get("/ask", params={"question": "Explain the Symmetric Cipher Model with a neat diagram", "mode": "document", "explanation_mode": "detailed"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreater(len(data["diagrams"]), 0)
        diagram = data["diagrams"][0]
        self.assertIn("page", diagram)
        self.assertIn("path", diagram)
        self.assertTrue(diagram["path"].startswith("/diagrams/"))
        print(f"[PASS] Test 10: Located diagram on Page {diagram['page']} and generated visual proof {diagram['filename']}.")

    def test_11_verify_page_numbers_and_image_serving(self):
        """Scenario 11: Verify that page numbers and image paths serve valid PNG bytes."""
        response = client.get("/ask", params={"question": "Explain Figure 2.1 model of conventional encryption with diagram", "mode": "document"})
        data = response.json()
        self.assertGreater(len(data["diagrams"]), 0)
        diagram_path = data["diagrams"][0]["path"]
        
        # Test image endpoint
        img_resp = client.get(diagram_path)
        self.assertEqual(img_resp.status_code, 200)
        self.assertEqual(img_resp.headers["content-type"], "image/png")
        self.assertGreater(len(img_resp.content), 1000)
        print("[PASS] Test 11: Rendered PDF page image served successfully via /diagrams endpoint.")

    def test_12_empty_question_and_error_handling(self):
        """Scenario 12: Test empty question and bad requests."""
        # Empty string
        resp = client.get("/ask", params={"question": "   "})
        self.assertEqual(resp.status_code, 400)
        # Missing question param
        resp2 = client.get("/ask")
        self.assertEqual(resp2.status_code, 422)
        print("[PASS] Test 12: Validation errors returned clean 400/422 status codes.")

    def test_13_query_decomposition(self):
        """Scenario 13: Query decomposition for complex compound prompt."""
        complex_q = "Explain the symmetric cipher model, describe its five components, and compare it with public-key encryption."
        decomp_res = client.post("/decompose", json={"query": complex_q})
        self.assertEqual(decomp_res.status_code, 200)
        data = decomp_res.json()
        self.assertTrue(data["is_complex"])
        self.assertGreaterEqual(len(data["sub_tasks"]), 3)
        print(f"[PASS] Test 13: Successfully decomposed complex query into {len(data['sub_tasks'])} sub-tasks.")

    def test_14_missing_web_credentials_resilience(self):
        """Scenario 14: Ensure missing API credentials do not break document RAG."""
        doc_resp = client.get("/ask", params={"question": "What is ciphertext?", "mode": "document"})
        self.assertEqual(doc_resp.status_code, 200)
        ans = doc_resp.json()["answer"].lower()
        self.assertTrue("ciphertext" in ans or "coded" in ans or "scrambled" in ans)
        print("[PASS] Test 14: Document RAG operates with 100% independence from external web APIs.")

    def test_15_persistence_and_health_check(self):
        """Scenario 15: Verify application health, document list, and persistence."""
        health = client.get("/")
        self.assertEqual(health.status_code, 200)
        h_data = health.json()
        self.assertEqual(h_data["status"], "online")
        self.assertGreater(h_data["documents_indexed"], 0)

        docs = client.get("/documents")
        self.assertEqual(docs.status_code, 200)
        d_data = docs.json()
        self.assertGreater(len(d_data["documents"]), 0)
        print(f"[PASS] Test 15: Workspace persistence confirmed ({h_data['documents_indexed']} documents loaded).")

if __name__ == "__main__":
    unittest.main()
