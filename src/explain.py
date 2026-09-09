from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import InputXGradient, IntegratedGradients, NoiseTunnel
import torch
import pandas as pd
import model_dataset
from tqdm import tqdm
import traceback

model_path = "outputs/models/bert"


#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, attn_implementation="eager")
model.to(device)
model.eval()

word_embeddings = model.bert.embeddings.word_embeddings

def forward_func(embeds, mask):
    return model(inputs_embeds=embeds, attention_mask=mask).logits

ixg = InputXGradient(forward_func)
ig = IntegratedGradients(forward_func)
nt = NoiseTunnel(ixg)


def compute_attributions(text: str, sentence_id, true_label=None) -> pd.DataFrame:
    """Compute IxG, IG, and SmoothGrad attributions for one sentence.

    Returns a long-format DataFrame with one row per (sub-word) token.
    """
    encoding = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    inputs_embeds = word_embeddings(input_ids)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.unk_token_id

    pad_id_tensor = torch.tensor([[pad_token_id]], device=device)
    pad_embed_vector = word_embeddings(pad_id_tensor)  # Shape: [1, 1, hidden_dim]

    # Expand pad embedding across sequence length
    baseline_embeds = pad_embed_vector.expand_as(inputs_embeds).clone()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred_class = outputs.logits.argmax(dim=1).item()
    
    # Helper function for L2 Norm Vector Aggregation
    def aggregate_attributions(attr_tensor):
        # Shape: [1, seq_len, hidden_dim] -> [seq_len]
        return torch.norm(attr_tensor, p=2, dim=-1).squeeze(0)
   
    attributions_ixg = ixg.attribute(
        inputs=inputs_embeds,
        additional_forward_args=(attention_mask,),
        target=pred_class,
    )
    scores_ixg = aggregate_attributions(attributions_ixg)
 
    attributions_ig = ig.attribute(
        inputs=inputs_embeds,
        baselines=baseline_embeds,
        additional_forward_args=(attention_mask,),
        target=pred_class,
        n_steps=150,
        internal_batch_size=8
    )
    scores_ig = aggregate_attributions(attributions_ig)
 
    attributions_sg = nt.attribute(
        inputs=inputs_embeds,
        additional_forward_args=(attention_mask,),
        target=pred_class,
        nt_type="smoothgrad",
        nt_samples=35,
        stdevs=0.15,
    )
    scores_sg = aggregate_attributions(attributions_sg)

    n_tokens = input_ids.shape[1]
    scores_random = torch.randn(n_tokens)
    scores_uniform = torch.ones(n_tokens) / n_tokens

    with torch.no_grad():
        attn_outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, output_attentions=True
        )
    last_layer_attn = attn_outputs.attentions[-1]
    scores_attention = last_layer_attn[0, :, 0, :].mean(dim=0).cpu()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return pd.DataFrame({
        "sentence_id": sentence_id,
        "text": text,
        "true_label": true_label,
        "token": tokens,
        "prediction": pred_class,
        "input_x_gradient": scores_ixg.detach().cpu().numpy(),
        "integrated_gradients": scores_ig.detach().cpu().numpy(),
        "smooth_grad": scores_sg.detach().cpu().numpy(),
        "random_attribution": scores_random.numpy(),
        "uniform_attribution": scores_uniform.numpy(),
        "attention_attribution": scores_attention.numpy(),
    })
    
    
    
holdout_dataset = model_dataset.load_holdout_split(n_train_used=30000)
subset = holdout_dataset.select(range(5))

if __name__ == "__main__":
  all_results = []
  failed = []

  # Using enumerate ensures `idx` is always defined for error logging
  for idx, row in enumerate(tqdm(subset, desc="Computing attributions")):
    # Handle missing keys dynamically for both IMDB ('text') and SST-2 ('sentence', 'idx')
    text = row.get("text") or row.get("sentence")
    row_id = row.get("idx", idx)
    label = row.get("label")

    try:
      result = compute_attributions(
          text=text,
          sentence_id=row_id,
          true_label=label,
      )
      all_results.append(result)
    except Exception as e:
      failed.append((row_id, text, str(e)))
      if len(failed) <= 3:
        print(f"\n--- Failure on row {row_id} ---")
        traceback.print_exc()
      continue

    if len(all_results) % 20 == 0:
      pd.concat(all_results, ignore_index=True).to_csv(
          "outputs/attribution_results_partial.csv", index=False
      )

  if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("outputs/attribution_results.csv", index=False)
    print(
        f"Done: {len(all_results)} sentences succeeded, {len(failed)} failed."
    )

  if failed:
    pd.DataFrame(failed, columns=["sentence_id", "text", "error"]).to_csv(
        "outputs/attribution_failures.csv", index=False
    )
    print("See attribution_failures.csv for failure details.")