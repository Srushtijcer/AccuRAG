import os
import sys
import unittest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(__file__))
from main import app

client = TestClient(app)

class TestConversationalSuite(unittest.TestCase):
    def test_01_feistel_initial_query(self):
        conv_id = "test_conv_feistel_1"
        res = client.post("/ask", json={
            "question": "Explain Classical Feistel Network with a neat diagram.",
            "mode": "document",
            "explanation_mode": "detailed",
            "conversation_id": conv_id
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        print("\n[TEST 1] Initial Feistel Query:")
        print("Answer length:", len(data["answer"]))
        print("Diagrams count:", len(data["diagrams"]))
        print("Format:", data.get("answer_format"))
        print("Coverage:", data.get("evidence_coverage"))
        self.assertIn("feistel", data["answer"].lower())
        self.assertGreater(len(data["diagrams"]), 0)
        self.assertEqual(data["diagrams"][0]["page"], 29)
        self.assertIn(data["verification_status"], ["Supported by document evidence", "Supported by evidence"])

    def test_02_followup_give_in_points(self):
        conv_id = "test_conv_feistel_1"
        # Follow up on same conversation_id
        res = client.post("/ask", json={
            "question": "Give it in points.",
            "mode": "document",
            "conversation_id": conv_id
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        print("\n[TEST 2] Follow-up 'Give it in points':")
        print("Format:", data.get("answer_format"))
        print("Answer preview:\n", data["answer"][:300])
        self.assertEqual(data.get("answer_format"), "bullet_points")
        self.assertIn("1.", data["answer"])
        self.assertGreater(len(data["diagrams"]), 0)
        self.assertEqual(data["diagrams"][0]["page"], 29)

    def test_03_followup_simple_words(self):
        conv_id = "test_conv_feistel_1"
        res = client.post("/ask", json={
            "question": "Explain it in simple words.",
            "mode": "document",
            "conversation_id": conv_id
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        print("\n[TEST 3] Follow-up 'Explain it in simple words':")
        print("Format:", data.get("answer_format"))
        print("Answer preview:\n", data["answer"][:300])
        self.assertEqual(data.get("answer_format"), "simple_words")
        self.assertIn("simple", data["answer"].lower())

    def test_04_followup_5_marks(self):
        conv_id = "test_conv_feistel_1"
        res = client.post("/ask", json={
            "question": "Make it a 5-mark answer.",
            "mode": "document",
            "conversation_id": conv_id
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        print("\n[TEST 4] Follow-up 'Make it a 5-mark answer':")
        print("Format:", data.get("answer_format"))
        print("Answer preview:\n", data["answer"][:300])
        self.assertEqual(data.get("answer_format"), "exam_answer")
        self.assertIn("5-mark", data["answer"].lower())

    def test_05_followup_diagram_only(self):
        conv_id = "test_conv_feistel_1"
        res = client.post("/ask", json={
            "question": "Explain only the diagram.",
            "mode": "document",
            "conversation_id": conv_id
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        print("\n[TEST 5] Follow-up 'Explain only the diagram':")
        print("Format:", data.get("answer_format"))
        print("Answer preview:\n", data["answer"][:300])
        self.assertEqual(data.get("answer_format"), "diagram_explanation")
        self.assertIn("diagram", data["answer"].lower())

    def test_06_followup_table(self):
        conv_id = "test_conv_feistel_1"
        res = client.post("/ask", json={
            "question": "Give the answer in a table.",
            "mode": "document",
            "conversation_id": conv_id
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        print("\n[TEST 6] Follow-up 'Give the answer in a table':")
        print("Format:", data.get("answer_format"))
        print("Answer preview:\n", data["answer"][:300])
        self.assertEqual(data.get("answer_format"), "comparison_table")
        self.assertIn("|", data["answer"])

    def test_07_difference_question_table(self):
        res = client.post("/ask", json={
            "question": "Differentiate between symmetric and asymmetric encryption.",
            "mode": "document",
            "conversation_id": "test_conv_diff"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        print("\n[TEST 7] Difference Question with Comparison Table:")
        print("Format:", data.get("answer_format"))
        self.assertEqual(data.get("answer_format"), "comparison_table")
        self.assertIn("|", data["answer"])
        self.assertIn("symmetric", data["answer"].lower())

    def test_08_empty_context_followup_clarification(self):
        # A completely fresh conversation ID with no previous query
        res = client.post("/ask", json={
            "question": "Give it in points.",
            "conversation_id": "brand_new_empty_conv"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        print("\n[TEST 8] Empty Context Followup Clarification:")
        print("Answer:", data["answer"])
        self.assertEqual(data.get("answer_format"), "clarification")
        self.assertIn("initial question", data["answer"].lower())

if __name__ == "__main__":
    unittest.main()
