"""
Model utilities for ASL Gloss to English translation.
Provides helper functions for model initialization, training, and evaluation.
"""

import torch
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import evaluate
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASLGlossDataset:
    """Custom dataset class for ASL gloss to English translation."""
    
    def __init__(self, data: List[Dict], tokenizer: T5Tokenizer, max_length: int = 128):
        """
        Initialize dataset.
        
        Args:
            data: List of dictionaries with 'input' and 'target' keys
            tokenizer: T5 tokenizer instance
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Add task prefix for T5
        input_text = f"translate ASL gloss to English: {item['input']}"
        target_text = item['target']
        
        # Tokenize inputs
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize targets
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length', 
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

def prepare_datasets(train_data: List[Dict], val_data: List[Dict], 
                    tokenizer: T5Tokenizer, max_length: int = 128) -> Tuple[Dataset, Dataset]:
    """
    Prepare HuggingFace datasets from raw data.
    
    Args:
        train_data: Training data list
        val_data: Validation data list
        tokenizer: T5 tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    def preprocess_function(examples):
        # Add task prefix for T5
        inputs = [f"translate ASL gloss to English: {inp}" for inp in examples['input']]
        targets = examples['target']
        
        # Tokenize inputs
        model_inputs = tokenizer(
            inputs, 
            max_length=max_length, 
            padding='max_length',
            truncation=True
        )
        
        # Tokenize targets
        labels = tokenizer(
            targets, 
            max_length=max_length, 
            padding='max_length',
            truncation=True
        )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Apply preprocessing
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)
    
    return train_dataset, val_dataset

def load_model_and_tokenizer(model_name: str = "distilbert/distilt5-base-uncased") -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
    """
    Load DistilT5 model and tokenizer.
    
    Args:
        model_name: Model name or path
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # Add special tokens if needed
    special_tokens = ["<gloss>", "</gloss>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    # Load model
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Resize token embeddings if we added special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    logger.info(f"Model loaded successfully. Vocab size: {len(tokenizer)}")
    
    return model, tokenizer

class ASLTranslationTrainer:
    """Custom trainer class for ASL gloss translation."""
    
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer,
                 train_dataset: Dataset, val_dataset: Dataset, 
                 output_dir: str = "./results"):
        """
        Initialize trainer.
        
        Args:
            model: T5 model instance
            tokenizer: T5 tokenizer
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Output directory for saved models
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        
        # Load evaluation metrics
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")
        
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics (BLEU, ROUGE)."""
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in labels (used for padding)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Clean up text
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]  # BLEU expects list of references
        
        # Compute BLEU score
        bleu_result = self.bleu_metric.compute(
            predictions=decoded_preds, 
            references=decoded_labels
        )
        
        # Compute ROUGE scores
        rouge_result = self.rouge_metric.compute(
            predictions=decoded_preds,
            references=[ref[0] for ref in decoded_labels]
        )
        
        return {
            "bleu": bleu_result["bleu"],
            "rouge1": rouge_result["rouge1"],
            "rouge2": rouge_result["rouge2"], 
            "rougeL": rouge_result["rougeL"]
        }
    
    def create_trainer(self, **kwargs) -> Trainer:
        """Create HuggingFace Trainer instance."""
        
        # Default training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=500,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=False,
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True,
            **kwargs
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        return trainer

def save_model_and_tokenizer(model: T5ForConditionalGeneration, tokenizer: T5Tokenizer, 
                            output_path: str):
    """
    Save model and tokenizer to specified path.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        output_path: Output directory path
    """
    logger.info(f"Saving model to {output_path}")
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info("Model saved successfully!")

def load_trained_model(model_path: str) -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
    """
    Load a trained model and tokenizer.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading trained model from {model_path}")
    
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    logger.info("Model loaded successfully!")
    
    return model, tokenizer

def calculate_model_size(model: T5ForConditionalGeneration) -> Dict[str, float]:
    """
    Calculate model size metrics.
    
    Args:
        model: T5 model instance
        
    Returns:
        Dictionary with size metrics
    """
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate size in MB (assuming float32)
    size_mb = param_count * 4 / (1024 * 1024)
    
    return {
        "total_params": param_count,
        "trainable_params": trainable_params,
        "size_mb": round(size_mb, 2)
    }

def test_model_inference(model: T5ForConditionalGeneration, tokenizer: T5Tokenizer,
                        test_inputs: List[str], device: str = "cpu") -> List[str]:
    """
    Test model inference on sample inputs.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_inputs: List of gloss inputs to test
        device: Device to run inference on
        
    Returns:
        List of generated English translations
    """
    model.eval()
    model.to(device)
    
    translations = []
    
    with torch.no_grad():
        for gloss_input in test_inputs:
            # Add task prefix
            input_text = f"translate ASL gloss to English: {gloss_input}"
            
            # Tokenize
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=128,
                padding=True,
                truncation=True
            ).to(device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
            
            # Decode
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translations.append(translation.strip())
    
    return translations