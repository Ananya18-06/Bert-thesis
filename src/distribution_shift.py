
import random
import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    MarianTokenizer,
    MarianMTModel,
)
from sentence_transformers import SentenceTransformer, util



# Setup

model_path = "outputs/models/bert"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()


NOISE_INTENSITIES = {"n1": 0.05, "n2": 0.10, "n3": 0.20}
SBERT_THRESHOLD = 0.85


class CovariateShiftPerturber:

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.keyboard_neighbors = {
            'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfxc', 'e': 'wsdr',
            'f': 'rtgvcd', 'g': 'tyhbvf', 'h': 'yujnbg', 'i': 'ujko', 'j': 'uikmnh',
            'k': 'ijlm', 'l': 'okp', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
            'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'wedxza', 't': 'rfgy',
            'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
            'z': 'asx'
        }

    def character_level_perturbation(self, text:str, rho:float) -> str:
        if not text or rho <= 0.0:
            return text

        words = text.split()
        perturbed_words = []

        for word in words:
            if len(word) <= 2:
                perturbed_words.append(word)
                continue

            no_chars_to_perturb = max(1, int(round(len(word) * rho)))
            char_list = list(word)
            indices_to_perturb = random.sample(
                range(len(char_list)), min(no_chars_to_perturb, len(char_list))
            )

            for idx in indices_to_perturb:
                orig_char = char_list[idx].lower()
                mutation_type = random.choice(['substitute', 'swap', 'delete'])

                if mutation_type == 'substitute' and orig_char in self.keyboard_neighbors:
                    char_list[idx] = random.choice(self.keyboard_neighbors[orig_char])
                elif mutation_type == 'swap' and idx < len(char_list) - 1:
                    char_list[idx], char_list[idx + 1] = char_list[idx + 1], char_list[idx]
                elif mutation_type == 'delete' and len(char_list) > 3:
                    char_list.pop(idx)    # len of input changes, alignment required
                    break

            perturbed_words.append("".join(char_list))

        return " ".join(perturbed_words)


class SBERTFilteredBackTranslator:

    def __init__(self,
                 en_de_model_name: str = "Helsinki-NLP/opus-mt-en-de",
                 de_en_model_name: str = "Helsinki-NLP/opus-mt-de-en",
                 sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):

        print("Loading translation models and tokenizers...")
        self.device = device

        self.en2de_tokenizer = MarianTokenizer.from_pretrained(en_de_model_name)
        self.en2de_model = MarianMTModel.from_pretrained(en_de_model_name).to(self.device)
        
        self.de2en_tokenizer = MarianTokenizer.from_pretrained(de_en_model_name)
        self.de2en_model = MarianMTModel.from_pretrained(de_en_model_name).to(self.device)

        print("Loading Sentence-BERT filter model...")
        self.sbert = SentenceTransformer(sbert_model_name)

    def _translate(self, text: str, model, tokenizer) -> str:
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs)
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def generate_paraphrase(self, text: str, similarity_threshold: float = SBERT_THRESHOLD):
        if not text.strip():
            return {"original": text, "paraphrase": "", "similarity": 0.0, "passed_filter": False}

        de_translation = self._translate(text, self.en2de_model, self.en2de_tokenizer)
        en_paraphrase = self._translate(de_translation, self.de2en_model, self.de2en_tokenizer)
        #cosine similarity computed on embeddings from SBERT specifically trained to make cosine similarity meningful for sentences
        embeddings = self.sbert.encode([text, en_paraphrase], convert_to_tensor=True)
        cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item() #cosine similarity between SBERT embeddings
        passed = cosine_sim > similarity_threshold

        return {
            "original": text,
            "intermediate_de": de_translation,
            "paraphrase": en_paraphrase,
            "similarity": round(cosine_sim, 4),
            "passed_filter": passed,
        }



# Prediction helper
def predict(text: str) -> int:
    encoding = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = model(**encoding).logits
    return logits.argmax(dim=1).item()



def perturb_and_check(sentence_id, text, original_prediction, perturber, back_translator):
    rows = []

    for ptype, rho in NOISE_INTENSITIES.items():
        perturbed_text = perturber.character_level_perturbation(text, rho)
        perturbed_prediction = predict(perturbed_text)
        prediction_changed = perturbed_prediction != original_prediction

        rows.append({
            "sentence_id": sentence_id,
            "perturbation_type": ptype,
            "rho": rho,
            "original_text": text,
            "perturbed_text": perturbed_text,
            "sbert_similarity": None,
            "sbert_passed": None,
            "original_prediction": original_prediction,
            "perturbed_prediction": perturbed_prediction,
            "prediction_changed": prediction_changed,
            # excluded if the prediction flipped (methodology Setup step 5)
            "excluded": prediction_changed,
        })

    bt_result = back_translator.generate_paraphrase(text, similarity_threshold=SBERT_THRESHOLD)
    bt_prediction = predict(bt_result["paraphrase"]) if bt_result["paraphrase"] else original_prediction
    bt_prediction_changed = bt_prediction != original_prediction

    rows.append({
        "sentence_id": sentence_id,
        "perturbation_type": "bt",
        "rho": None,
        "original_text": text,
        "perturbed_text": bt_result["paraphrase"],
        "sbert_similarity": bt_result["similarity"],
        "sbert_passed": bt_result["passed_filter"],
        "original_prediction": original_prediction,
        "perturbed_prediction": bt_prediction,
        "prediction_changed": bt_prediction_changed,
        # excluded if EITHER the prediction flipped OR the SBERT filter
        # rejected the paraphrase as too semantically different
        "excluded": bt_prediction_changed or not bt_result["passed_filter"],
    })

    return rows



attr_df = pd.read_csv("outputs/attribution_results.csv")
sentences = (
    attr_df[["sentence_id", "text", "prediction"]]
    .drop_duplicates(subset="sentence_id")
    .reset_index(drop=True)
)

perturber = CovariateShiftPerturber(seed=42)
back_translator = SBERTFilteredBackTranslator()

all_rows = []
failed = []

for _, row in tqdm(sentences.iterrows(), total=len(sentences), desc="Generating perturbations"):
    try:
        result_rows = perturb_and_check(
            sentence_id=row["sentence_id"],
            text=row["text"],
            original_prediction=int(row["prediction"]),
            perturber=perturber,
            back_translator=back_translator,
        )
        all_rows.extend(result_rows)
    except Exception as e:
        failed.append((row["sentence_id"], row["text"], str(e)))
        continue

    if len(all_rows) % 40 == 0:  # ~10 sentences x 4 perturbation types
        pd.DataFrame(all_rows).to_csv(
            "outputs/perturbations_partial.csv", index=False
        )

results_df = pd.DataFrame(all_rows)
results_df.to_csv("outputs/perturbations.csv", index=False)

print(f"\nDone: {len(sentences) - len(failed)} sentences succeeded, {len(failed)} failed.")
if failed:
    pd.DataFrame(failed, columns=["sentence_id", "text", "error"]).to_csv(
        "outputs/perturbation_failures.csv", index=False
    )
    print("See outputs/perturbation_failures.csv for details.")

print("\nExclusion rate by perturbation type:")
print(results_df.groupby("perturbation_type")["excluded"].mean())