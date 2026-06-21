from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import InputXGradient
from captum.attr import LayerIntegratedGradients
from captum.attr import NoiseTunnel
import torch
import pandas as pd

model_path = "outputs/models/bert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

text = "The movie was surprisingly good and I enjoyed it."

encoding = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True
)

input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]
inputs_embeds = model.bert.embeddings(input_ids)

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask
)

pred_class = outputs.logits.argmax(dim=1).item()


def forward_func(inputs_embeds, attention_mask):
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask = attention_mask
    )
    return outputs.logits


#------------INPUT * GRADIENT------------------- 
# This computes (Input * ∂y/x) for every token
ixg = InputXGradient(forward_func)

attributions2 = ixg.attribute(
    inputs = inputs_embeds,
    additional_forward_args = (attention_mask, ),
    target = pred_class
)
print(attributions2.shape)
scores2 = attributions2.sum(dim = -1). squeeze(0)

#------------INTEGRATED GRADIENT------------------- 
# calculates: (x -x')∫ (∂F (x' + α(x - x')) / ∂x), integrated over interval 0 to 1
lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)
inputs_embeds = model.bert.embeddings(input_ids)
attributions = lig.attribute(
    inputs=inputs_embeds,
    additional_forward_args = (attention_mask, ),
    target = pred_class,
    n_steps = 50
)
print(attributions.shape)
scores = attributions.sum(dim = -1). squeeze(0) #convert attribution vector to one score per token


#------------SMOOTH GRAD------------------- 
nt = NoiseTunnel(lig)
attributions1 = nt.attribute(
    inputs = inputs_embeds,
    additional_forward_args = (attention_mask,) ,
    target = pred_class,
    nt_type = 'smoothgrad',
    nt_samples = 20,
    stdevs = 0.1
)
print(attributions1.shape)
scores1 = attributions1.sum(dim = -1). squeeze(0)

#-----------------------------------------------------------------

#aggrigate attribution values
tokens = tokenizer.convert_ids_to_tokens(
    input_ids[0]
)

df = pd.DataFrame({
    "token": tokens,
    "prediction": pred_class,
    "Input * Gradient": scores2.detach().cpu().numpy(),
    "integrated gradient": scores.detach().cpu().numpy(),
    "smooth grad": scores1.detach().cpu().numpy()
})


#save to CVS
df.to_csv( 
    "integrated_gradients_results.csv",
    index=False
)
print(df)
