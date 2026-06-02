
from huggingface_hub import login
import yaml
import pandas
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# -------------- CONSTANTS: model -----------------
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Model choice:
model_name = config["model_name"]

output_dir = config["output_dir"]

device = torch.device("cpu")

model = None
tokenizer = None


# ---------------- FUNCTIONS ----------------

def hf_login():
    token = "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"
    login(token=token, add_to_git_credential=True)

def load_model():
    global model, tokenizer

    print(f"Using device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Model and tokenizer loaded successfully.")
    text = "Replace me by any text you'd like."
    
    encoded = tokenizer(text, return_tensors='pt')
    encoded = {k: v.to(device) for k, v in encoded.items()}
    output = model(**encoded)
    print(output.logits)
    return model, tokenizer
    


# ---------------- MAIN ENTRY POINT ----------------

if __name__ == "__main__":
    hf_login()
    load_model()
    