import tokenizer

if __name__ == "__main__":
    from train_gpt2_bpe import train_gpt2_bpe

    # 1. Entraînement initial
    corpus = "Le projet mini-GPT du laboratoire ECE Paris. Entraînement BPE."
    merges, vocab = train_gpt2_bpe(corpus, vocab_size=280)
    special_toks = {"<|endoftext|>": 280}

    tokenizer_original = tokenizer(merges, vocab, special_tokens=special_toks)

    # 2. Sauvegarde sur le disque
    model_path = "tokenizer/vocab_280.json"
    tokenizer_original.save(model_path)

    # 3. Rechargement instantané sans réentraîner
    tokenizer_recharge = tokenizer.load(model_path)

    # 4. Vérification de l'encodage
    sample = "Le projet mini-GPT <|endoftext|>"
    assert tokenizer_original.encode(sample) == tokenizer_recharge.encode(sample)
    print(" Validation : Le tokenizer rechargé produit exactement les mêmes tokens !")