from model import load_model
import yaml
from datasets import load_dataset

# -------------- CONSTANTS: dataset -----------------
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_name = config["dataset_name"]
print("CONFIG DATASET =", dataset_name)

# ---------------- FUNCTIONS ----------------

model, tokenizer = load_model()

def tokenize(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        padding=True,
        max_length=128
    )
    
def load_and_prepare_datasets():

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