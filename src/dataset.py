from model import tokenizer
from datasets import load_dataset

# -------------- CONSTANTS: dataset -----------------
dataset_name = "stanfordnlp/sst2"

# ---------------- FUNCTIONS ----------------


def tokenize(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        padding=True,
        max_length=128
    )
    
def load_and_prepare_datasets():
    global formatted_dataset, test_dataset, evaluation_dataset

    dataset = load_dataset(dataset_name)
    train_dataset = dataset["train"]
    test_dataset =  test_dataset = dataset["validation"]  # SST-2 uses validation as test
   
    # small subset for experiments (thesis-friendly)
    train_dataset = train_dataset.shuffle(seed=42).select(range(1000))
    test_dataset = test_dataset.shuffle(seed=42).select(range(200))
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)


    print("Dataset loaded successfully!")

    print("\nExample:")
    print(train_dataset[0])

    return train_dataset, test_dataset

# ---------------- MAIN ENTRY POINT ----------------

if __name__ == "__main__":
    load_and_prepare_datasets()