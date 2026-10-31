import os
import sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

from generation_engine import GenerationEngine

ge = GenerationEngine()
ctx = (
    "The Feistel cipher approximates an ideal block cipher using a product cipher. "
    "Inputs are a plaintext block of length 2w bits and a key K. The block is divided into two halves L0 and R0. "
    "Each round i has inputs Li-1 and Ri-1, and subkey Ki. Li = Ri-1, and Ri = Li-1 XOR F(Ri-1, Ki). "
    "The process repeats for n rounds. Decryption uses the same algorithm with subkeys in reverse order."
)

p1 = f"Provide a technical definition of the Feistel network based on this context:\n{ctx}"
inp1 = ge.tokenizer(p1, return_tensors="pt")
out1 = ge.model.generate(**inp1, max_new_tokens=150, num_beams=2)
print("DEFINITION:", ge._clean_output(ge.tokenizer.decode(out1[0], skip_special_tokens=True)))

p2 = f"List the working steps of the Feistel network based on this context:\n{ctx}"
inp2 = ge.tokenizer(p2, return_tensors="pt")
out2 = ge.model.generate(**inp2, max_new_tokens=200, num_beams=2)
print("WORKING:", ge._clean_output(ge.tokenizer.decode(out2[0], skip_special_tokens=True)))
