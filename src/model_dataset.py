from model import load_model
import yaml
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------- CONSTANTS: dataset  and model-----------------
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_name = config["dataset_name"]
print("CONFIG DATASET =", dataset_name)

model_name = config["model_name"]

output_dir = config["output_dir"]

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.mps.is_available() 
                      else "cpu")

model = None
tokenizer = None

# ---------------- FUNCTIONS ----------------

model, tokenizer = load_model()
def load_model():
    global model, tokenizer
    print(f"Using device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def tokenize(batch):
    return tokenizer(
        batch["sentence"],  # text for IMDB
        truncation=True,
        padding=True,
        max_length=128
    )
    
def load_and_prepare_datasets():

    dataset = load_dataset(dataset_name)
    train_dataset = dataset["train"]
    test_dataset  = dataset["validation"]  # IMDB uses "test" as test
   
    # small subset for experiments
    train_dataset = train_dataset.shuffle(seed=42).select(range(30000)) # IMDB has only 25k rows
    test_dataset = test_dataset.shuffle(seed=42).select(range(872))
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)


    print("Dataset loaded successfully!")

    print("\nExample:")
    print(train_dataset[0])

    return train_dataset, test_dataset

def load_holdout_split(n_train_used: int = 1000):
    
    dataset = load_dataset(dataset_name)
    train_dataset = dataset["train"].shuffle(seed=42)

    holdout_dataset = train_dataset.select(
        range(n_train_used, len(train_dataset))
    )
    print(f"Held-out pool: {len(holdout_dataset)} rows "
          f"(train split has {len(train_dataset)}, {n_train_used} used for training)")

    return holdout_dataset


# ---------------- MAIN ENTRY POINT ----------------

if __name__ == "__main__":
    load_and_prepare_datasets()
    