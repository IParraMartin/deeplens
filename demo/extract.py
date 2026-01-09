from deeplens.extractor import FromHuggingFace

extractor = FromHuggingFace(
    hf_model="microsoft/phi-2",
    layer=3,
    dataset_name="HuggingFaceFW/fineweb",
    num_samples=100,
    seq_length=128,
    inference_batch_size=16,
    device="auto",
    save_features=True
)

features = extractor.extract_features()