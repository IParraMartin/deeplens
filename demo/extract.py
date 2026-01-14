from deeplens.extractor import FromHuggingFace

extractor = FromHuggingFace(
    hf_model="gpt2",
    layer=1,
    dataset_name="HuggingFaceFW/fineweb",
    num_samples=10000,
    seq_length=1024,
    inference_batch_size=8,
    device="auto",
    save_features=True
)

features = extractor.extract_features_batched(chunk_size=1000)
