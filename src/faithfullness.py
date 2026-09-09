
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


model_path = "outputs/models/bert"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}
METHOD_COLUMNS = [
    "input_x_gradient", "integrated_gradients", "smooth_grad",
    "random_attribution", "uniform_attribution", "attention_attribution",
]


# Comprehensiveness / sufficiency curves for one sentence, one method

def faithfulness_curves(input_ids, attention_mask, sorted_indices, pred_class):
   
    pad_id = tokenizer.pad_token_id
    n = len(sorted_indices)

    comprehensiveness_scores = []  # confidence as top-k important tokens are REMOVED
    sufficiency_scores = []         # confidence as only top-k important tokens are KEPT

    for k in range(n + 1):
        removed = input_ids.clone()
        for idx in sorted_indices[:k]:
            removed[0, idx] = pad_id
        removed_mask = (removed != pad_id).long()
        with torch.no_grad():
            probs = torch.softmax(
                model(input_ids=removed, attention_mask=removed_mask).logits, dim=1
            )
        comprehensiveness_scores.append(probs[0, pred_class].item())

        # sufficiency: keep only the top-k important tokens ---
        kept = input_ids.clone()
        keep_set = set(sorted_indices[:k])
        for pos in range(input_ids.shape[1]):
            tok_id = input_ids[0, pos].item()
            if pos not in sorted_indices and tokenizer.convert_ids_to_tokens([tok_id])[0] in SPECIAL_TOKENS:
                continue
            if pos not in keep_set and pos in sorted_indices:
                kept[0, pos] = pad_id
        kept_mask = (kept != pad_id).long()
        with torch.no_grad():
            probs = torch.softmax(
                model(input_ids=kept, attention_mask=kept_mask).logits, dim=1
            )
        sufficiency_scores.append(probs[0, pred_class].item())

    return comprehensiveness_scores, sufficiency_scores


def aopc(scores, original_confidence):
    drops = [original_confidence - s for s in scores[1:]]
    return sum(drops) / len(drops) if drops else 0.0


def evaluate_faithfulness_for_sentence(text: str, pred_class: int, token_df: pd.DataFrame):
    encoding = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = encoding["input_ids"].to(device)                       # gives integer vocabulary for each token
    attention_mask = encoding["attention_mask"].to(device)             # 0 and 1 tensor marking which, inut id is content and which is padding
 
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])             # readable token for csv files

    valid_indices = [i for i, tok in enumerate(tokens) if tok not in SPECIAL_TOKENS]   # positions of relevant tokens 
 
    with torch.no_grad():      # disable gradient tracking, save memory and computation time
        orig_probs = torch.softmax(
            model(input_ids=input_ids, attention_mask=attention_mask).logits, dim=1
        )
    original_confidence = orig_probs[0, pred_class].item()
 
    rows = []
    for method in METHOD_COLUMNS:
        scores = token_df[method].values                         # raw attribution scores 
        importance = torch.tensor(scores).abs()                  # absolute value of attribution 
        valid_importance = importance[valid_indices]
        order = torch.argsort(valid_importance, descending=True) # arrange attribution score in descending order 
        sorted_indices = [valid_indices[i] for i in order]
 
        comp_scores, suff_scores = faithfulness_curves(
            input_ids, attention_mask, sorted_indices, pred_class
        )
 
        rows.append({
            "method": method,
            "original_confidence": original_confidence,
            "comprehensiveness_aopc": aopc(comp_scores, original_confidence),
            "sufficiency_aopc": aopc(suff_scores, original_confidence),
        })
    return rows



# Loop over every sentence and every method
if __name__ == "__main__":
    attr_df = pd.read_csv("outputs/attribution_results.csv")
    results = []
    failed = []

    sentence_ids = attr_df["sentence_id"].unique() #all sentence ids 

    for sid in tqdm(sentence_ids, desc="Computing faithfulness"):
        group = attr_df[attr_df["sentence_id"] == sid].reset_index(drop=True)
        text = group["text"].iloc[0]
        pred_class = int(group["prediction"].iloc[0])
        
        try:
            sentence_results = evaluate_faithfulness_for_sentence(text, pred_class, group)
            for res in sentence_results:
                res["sentence_id"] = sid
                results.append(res)
        except Exception as e:
            failed.append((sid, text, str(e)))


        if len(results) % 60 == 0:  # ~20 sentences x 3 methods
            pd.DataFrame(results).to_csv(
                "outputs/faithfulness_results_partial.csv", index=False
            )


    # Save
    print(f"\nDone: {len(sentence_ids) - len(failed)} sentences succeeded, {len(failed)} failed.")

    if failed:
        pd.DataFrame(failed, columns=["sentence_id", "text", "error"]).to_csv(
            "outputs/faithfulness_failures.csv", index=False
        )
        print("See outputs/faithfulness_failures.csv for details.")

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv("outputs/faithfulness_results.csv", index=False)
        print("Saved outputs/faithfulness_results.csv")
        print(results_df.groupby("method")[["comprehensiveness_aopc", "sufficiency_aopc"]].mean())
    else:
        print("No results to save.")


