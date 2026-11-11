import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gpt.model import GPT, GPTConfig
from tokenizer.tokenizer import Tokenizer
from data_pipeline.clean_text import clean_text
from data_pipeline.pii_masking import mask_pii


def test_tokenizer_roundtrip():
    tokenizer = Tokenizer.load("tokenizer/vocab_1000.json")
    text = "HARPAGON : Holà !"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert len(encoded) > 0
    assert "HARPAGON" in decoded
    print("✅ Test Tokenizer : OK")


def test_pipeline():
    sample = "Contactez-moi à test@example.com."
    cleaned = clean_text(sample)
    masked = mask_pii(cleaned)
    assert "<EMAIL>" in masked
    print("✅ Test Pipeline PII : OK")


def test_model_forward():
    config = GPTConfig(vocab_size=1001, block_size=128, n_layer=2, n_head=2, n_embd=120)
    model = GPT(config)
    idx = torch.randint(0, 1000, (2, 64))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 64, 1001)
    print("✅ Test Modèle Forward : OK")


if __name__ == "__main__":
    test_tokenizer_roundtrip()
    test_pipeline()
    test_model_forward()