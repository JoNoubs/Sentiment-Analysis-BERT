from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from .data_extraction import load_data
from .data_processing import tokenize_data

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    accuracy = (labels == preds).mean()
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def train_model():
    try:
        df = load_data()
        if len(df) < 5:
            raise ValueError("Dataset is too small for training. Need at least 5 samples.")
        test_size = max(0.2, 1 / len(df))
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
        if len(val_df) == 0:
            raise ValueError("Validation set is empty. Increase dataset size or adjust test_size.")
        train_encodings, train_labels = tokenize_data(train_df)
        val_encodings, val_labels = tokenize_data(val_df)
        train_dataset = [{"input_ids": torch.tensor(train_encodings['input_ids'][i]),
                          "attention_mask": torch.tensor(train_encodings['attention_mask'][i]),
                          "labels": torch.tensor(train_labels[i])} for i in range(len(train_labels))]
        val_dataset = [{"input_ids": torch.tensor(val_encodings['input_ids'][i]),
                        "attention_mask": torch.tensor(val_encodings['attention_mask'][i]),
                        "labels": torch.tensor(val_labels[i])} for i in range(len(val_labels))]
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
        training_args = TrainingArguments(
            output_dir="./models",
            num_train_epochs=2,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            logging_dir="./logs",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy"
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
        print("Model training completed and saved to ./models")
        return trainer
    except Exception as e:
        print(f"Error during model training: {e}")
        raise

def load_trained_model():
    try:
        model = AutoModelForSequenceClassification.from_pretrained("./models")
        print("Loaded pre-trained model from ./models")
        return model
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        raise