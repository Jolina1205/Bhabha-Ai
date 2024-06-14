import modal

stub = modal.Stub("llama2-translation-filter")

@stub.function(image=modal.Image.debian_slim().pip_install(["transformers", "datasets"]))
def infer():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import load_dataset, load_metric
    from torch.utils.data import DataLoader

    # Load the fine-tuned model
    model_name = "satpalsr/llama2-translation-filter-full"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the dataset
    dataset = load_dataset("satpalsr/chatml-translation-filter", split="validation")

    # Prepare the dataset for the model
    def preprocess(example):
        encoding = tokenizer(example['question'], example['answer'], truncation=True, padding=True)
        return encoding

    encoded_dataset = dataset.map(preprocess, batched=True)

    # Create a DataLoader
    batch_size = 16
    dataloader = DataLoader(encoded_dataset, batch_size=batch_size)

    # Initialize metric
    metric = load_metric("accuracy")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Run inference
    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    # Compute accuracy
    accuracy = metric.compute()
    print(f"Accuracy: {accuracy['accuracy']}")

if __name__ == "__main__":
    with stub.run():
        infer()
