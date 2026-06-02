import yaml
import numpy as np 
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

from model import load_model 
from dataset import load_and_prepare_datasets

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)
    accuracy =(predictions ==labels).mean()
    return{
        "accuracy": accuracy
    }

def main():
    with open("configs/config.yaml", "r") as f:
        config  =  yaml.safe_load(f)
        model, tokenizer = load_model()
        train_dataset, valid_dataset = load_and_prepare_datasets()
        
        data_collator = DataCollatorWithPadding(
            tokenizer = tokenizer
        )
        
        training_args = TrainingArguments(
            output_dir=config["output_dir"],
            learning_rate=config["lr"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["eval_size"],
            num_train_epochs=config["epochs"],
            weight_decay=config["weight_decay"],
            warmup_ratio=config["warmup_ratio"],
            gradient_accumulation_steps=config["gradient_accumulation"],
            max_grad_norm=config["max_grad_norm"],
            optim=config["optimizer"],
            lr_scheduler_type=config["scheduler"],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_dir="outputs/logs",
            seed=config["seed"],
            fp16=False, 
            push_to_hub=False
            
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[ 
                       EarlyStoppingCallback( 
                                             early_stopping_patience=5,
                                             early_stopping_threshold=0.01 
                                             ) 
                       ]
        )
        
        trainer.train()
        trainer.save_model(config["output_dir"])
        tokenizer.save_pretrained(config["output_dir"])
        
if __name__ == "__main__":
    main()

