import os
import shutil
from backend.model import train_model, compute_metrics, load_trained_model
from transformers import Trainer

def save_best_model():
    try:
        # Check if a pre-trained model already exists
        model_path = "models/model.safetensors"
        if os.path.exists(model_path):
            print("Pre-trained model found at models/model.safetensors, skipping training and evaluation.")
            return

        # If no pre-trained model, proceed with training
        trainer = train_model()
        metrics = trainer.evaluate()
        accuracy = metrics['eval_accuracy']
        precision = metrics.get('eval_precision', 0)
        recall = metrics.get('eval_recall', 0)
        f1 = metrics.get('eval_f1', 0)
        with open("metrics.txt", "w") as f:
            f.write(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\n")
        print("Best model saved in models/")
    except Exception as e:
        print(f"Error saving best model: {e}")
        raise

if __name__ == "__main__":
    save_best_model()