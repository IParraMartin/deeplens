from deeplens.extractor import FromHuggingFace

extractor = FromHuggingFace(
    hf_model="gpt2",
    layer=-1,
    dataset_name="HuggingFaceFW/fineweb",
    num_samples=2500,
    seq_length=1024,
    inference_batch_size=16,
    device="auto",
    save_features=True
)

features = extractor.extract_features()