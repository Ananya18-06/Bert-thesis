import torch
from model import load_model 
from dataset import load_and_prepare_datasets
device = torch.device("cpu")
model, tokenizer = load_model()
model.to(device)
model.eval()

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors = "pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).item()
    
    return predictions

if __name__ == "__main__":
    train, test = load_and_prepare_datasets()
    for i in range(10):
        sentence = test[i]["sentence"]
        label = test[i]["label"]
        
        pred = predict(sentence)
        print("MODEL CURRENTLY NOT FINE TUNED")
        print(f"Text: {sentence}")
        print(f"True value: {label}, Prediction: {pred}")
        print("-" * 40)


