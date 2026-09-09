"""
Final evaluation pipeline (Setup steps 6-13):
  6. Recompute attributions on perturbed (non-excluded) inputs
  7. Aggregate sub-word attributions to word level
  8. Align original <-> perturbed word-level attributions (SimAlign for
     back-translation; positional for character-noise, since word count
     and order are preserved by construction)
  9. Faithfulness per perturbation type
 10. Stability per method (answers RQ1)
 11. Faithfulness aggregation
 12/13. Correlate stability against faithfulness, per method (answers RQ2)
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from simalign import SentenceAligner

import explain          # exposes: model, tokenizer, device, compute_attributions()
import faithfullness    # exposes: evaluate_faithfulness_for_sentence(), METHOD_COLUMNS

METHODS = faithfullness.METHOD_COLUMNS  # ["input_x_gradient", "integrated_gradients", "smooth_grad"]
PERTURBATION_TYPES = ["n1", "n2", "n3", "bt"]
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}

aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="m")
# "m" = max-weight (optimal assignment) matching, matching the description
# in the methodology chapter.


# Step 7: word-level aggregation
def aggregate_to_word_level(token_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse sub-word (WordPiece) attributions to one row per word by
    summing scores of a word and its '##'-prefixed continuations.
    Special tokens ([CLS]/[SEP]/[PAD]) are dropped."""
    rows = []
    current_word, current_scores = None, {m: 0.0 for m in METHODS}

    def flush():
        if current_word is not None:
            row = {"word": current_word}
            row.update(current_scores)
            rows.append(row)

    for _, r in token_df.iterrows():
        tok = r["token"]
        if tok in SPECIAL_TOKENS:
            continue
        if tok.startswith("##"):
            current_word += tok[2:]
            for m in METHODS:
                current_scores[m] += r[m]
        else:
            flush()
            current_word = tok
            current_scores = {m: r[m] for m in METHODS}
    flush()

    return pd.DataFrame(rows)



# Step 8: alignment + stability for one (sentence, method, perturbation type)
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else np.nan


def aligned_vectors(orig_word_df, pert_word_df, method, ptype):
    #if ptype == "bt":
    src_words = orig_word_df["word"].tolist()
    trg_words = pert_word_df["word"].tolist()
    if not src_words or not trg_words:
        return None, None
    alignments = aligner.get_word_aligns(src_words, trg_words)["mwmf"]
    if len(alignments) < 2:
        return None, None
    orig_vec = np.array([orig_word_df.iloc[i][method] for i, j in alignments])
    pert_vec = np.array([pert_word_df.iloc[j][method] for i, j in alignments])
    return orig_vec, pert_vec
    #else:
        # Character-level noise: word count/order preserved by construction.
        #if len(orig_word_df) != len(pert_word_df):
            #return None, None
        #return orig_word_df[method].values, pert_word_df[method].values


def compute_stability(orig_vec, pert_vec):
    if orig_vec is None or len(orig_vec) < 2:
        return np.nan, np.nan
    cos = cosine_sim(orig_vec, pert_vec)
    if np.std(orig_vec) == 0 or np.std(pert_vec) == 0:
        if method !=  "uniform_attribution":
            print(f"WARNING:constant vector for non uniform method '{method} ")
    rank = spearmanr(orig_vec, pert_vec).correlation
    return cos, rank



# Main pipeline
attr_df = pd.read_csv("outputs/attribution_results.csv")
perturb_df = pd.read_csv("outputs/perturbations.csv")
perturb_df = perturb_df[~perturb_df["excluded"]].reset_index(drop=True)

orig_faith_df = pd.read_csv("outputs/faithfulness_results.csv")
orig_faith_lookup = {
    (row["sentence_id"], row["method"]): row
    for _, row in orig_faith_df.iterrows()
}



stability_rows = []
stability_per_type_rows = []
faithfulness_rows = []
failed = []

sentence_ids = attr_df["sentence_id"].unique()

for sid in tqdm(sentence_ids, desc="Evaluating stability & faithfulness"):
    
    #Aggrigate attribution of original input to word level
    orig_group = attr_df[attr_df["sentence_id"] == sid].reset_index(drop=True)
    orig_text = orig_group["text"].iloc[0]
    orig_pred = int(orig_group["prediction"].iloc[0])
    orig_word_df = aggregate_to_word_level(orig_group)

    # Faithfulness on the original (unperturbed) input, per method
    orig_faith = {}
    missing = False
    for method in METHODS:
        key = (sid, method)
        if key not in orig_faith_lookup:
            missing = True
            break
        orig_faith[method] = orig_faith_lookup[key]
    if missing:
        failed.append((sid, orig_text, "missing entry in faithfulness_results.csv for this sentence/method"))
        continue

    per_type_faith = {m: [] for m in METHODS}     # collects F_m^(k) across k, for aggregation
    per_type_stability = {m: {"cos": [], "rank": []} for m in METHODS}

    for ptype in PERTURBATION_TYPES:
        prow = perturb_df[(perturb_df["sentence_id"] == sid) & (perturb_df["perturbation_type"] == ptype)]
        if prow.empty:
            continue  # excluded or missing for this sentence/type
        prow = prow.iloc[0]
        pert_text = prow["perturbed_text"]
        pert_pred = int(prow["perturbed_prediction"])

        try:
            pert_token_df = explain.compute_attributions(pert_text, sentence_id=sid, true_label=pert_pred)
        except Exception as e:
            failed.append((sid, pert_text, f"attribution ({ptype}): {e}"))
            continue
        
        #Aggrigate attribution od perturbsted type
        pert_word_df = aggregate_to_word_level(pert_token_df)

        # Faithfulness on this perturbation, per method
        try:
            pert_faith_rows = faithfullness.evaluate_faithfulness_for_sentence(
                pert_text, pert_pred, pert_token_df
            )
            for r in pert_faith_rows:
                per_type_faith[r["method"]].append(r["comprehensiveness_aopc"])
                faithfulness_rows.append({
                    "sentence_id": sid, "method": r["method"], "perturbation_type": ptype,
                    "comprehensiveness_aopc": r["comprehensiveness_aopc"],
                    "sufficiency_aopc": r["sufficiency_aopc"],
                })
        except Exception as e:
            failed.append((sid, pert_text, f"faithfulness ({ptype}): {e}"))

        # Stability, per method
        for method in METHODS:
            orig_vec, pert_vec = aligned_vectors(orig_word_df, pert_word_df, method, ptype)
            cos, rank = compute_stability(orig_vec, pert_vec)
            per_type_stability[method]["cos"].append(cos)
            per_type_stability[method]["rank"].append(rank)
            stability_per_type_rows.append({
                "sentence_id": sid, "method": method, "perturbation_type": ptype,
                "S_cos": cos, "S_rank": rank,
            })

    # Step 10: stability averaged across the 4 perturbation types -- RQ1
    # Step 11: faithfulness aggregation -- F_final = (F_pert + F_orig) / 2
    for method in METHODS:
        cos_vals = [v for v in per_type_stability[method]["cos"] if not np.isnan(v)]
        rank_vals = [v for v in per_type_stability[method]["rank"] if not np.isnan(v)]
        S_cos = np.mean(cos_vals) if cos_vals else np.nan
        S_rank = np.mean(rank_vals) if rank_vals else np.nan

        F_pert = np.mean(per_type_faith[method]) if per_type_faith[method] else np.nan
        F_orig = orig_faith[method]["comprehensiveness_aopc"]
        F_final = np.nanmean([F_pert, F_orig])

        stability_rows.append({
            "sentence_id": sid,
            "method": method,
            "S_cos": S_cos,
            "S_rank": S_rank,
            "F_orig": F_orig,
            "F_pert": F_pert,
            "F_final": F_final,
        })

# ---------------------------------------------------------------------------
# Save intermediate results
# ---------------------------------------------------------------------------
stability_df = pd.DataFrame(stability_rows)
stability_df.to_csv("outputs/stability_faithfulness_per_sentence.csv", index=False)


stability_per_type_df = pd.DataFrame(stability_per_type_rows)
stability_per_type_df.to_csv("outputs/stability_per_perturbation_type.csv", index=False)

faithfulness_perturbed_df = pd.DataFrame(faithfulness_rows)
faithfulness_perturbed_df.to_csv("outputs/faithfulness_perturbed.csv", index=False)

if failed:
    pd.DataFrame(failed, columns=["sentence_id", "text", "error"]).to_csv(
        "outputs/evaluation_failures.csv", index=False
    )
    print(f"{len(failed)} failures logged to outputs/evaluation_failures.csv")

# ---------------------------------------------------------------------------
# Steps 12/13: correlate stability against final faithfulness -- RQ2
# ---------------------------------------------------------------------------
print("\n=== Mean stability and faithfulness per method ===")
print(stability_df.groupby("method")[["S_cos", "S_rank", "F_orig", "F_pert", "F_final"]].mean())

print("\n=== Stability-faithfulness correlation per method (RQ2) ===")
correlation_rows = []
for method in METHODS:
    sub = stability_df[stability_df["method"] == method].dropna(subset=["S_cos", "S_rank", "F_final"])
    rho_cos = sub["S_cos"].corr(sub["F_final"])
    rho_rank = sub["S_rank"].corr(sub["F_final"])
    correlation_rows.append({"method": method, "rho_cos": rho_cos, "rho_rank": rho_rank, "n": len(sub)})

correlation_df = pd.DataFrame(correlation_rows)
correlation_df.to_csv("outputs/stability_faithfulness_correlation.csv", index=False)
print(correlation_df)

    