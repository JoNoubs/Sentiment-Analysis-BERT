# src/model.py
# Purpose: Train a BERT model for sentiment analysis with 3 labels
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from src.data_extraction import load_data
from src.data_processing import tokenize_data, split_data

def compute_metrics(pred):
    # Compute accuracy metric for evaluation
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": (labels == preds).mean()}

def train_model():
    # Load and split data
    df = load_data()
    train_df, val_df = split_data(df)
    train_encodings, train_labels = tokenize_data(train_df)
    val_encodings, val_labels = tokenize_data(val_df)

    # Prepare datasets
    train_dataset = [{"input_ids": torch.tensor(train_encodings['input_ids'][i]),
                      "attention_mask": torch.tensor(train_encodings['attention_mask'][i]),
                      "labels": torch.tensor(train_labels[i])} for i in range(len(train_labels))]
    val_dataset = [{"input_ids": torch.tensor(val_encodings['input_ids'][i]),
                    "attention_mask": torch.tensor(val_encodings['attention_mask'][i]),
                    "labels": torch.tensor(val_labels[i])} for i in range(len(val_labels))]

    # Initialize model and trainer
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    training_args = TrainingArguments(
        output_dir="./models",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        logging_dir="./logs",
        evaluation_strategy="epoch"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    model.save_pretrained("./models")

if __name__ == "__main__":
    # Execute training
    train_model()
