"""
Training script for ASL Gloss to English translation model.
Fine-tunes DistilT5 on synthetic gloss-to-English dataset.
"""

import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
from transformers import set_seed

from model_utils import (
    load_model_and_tokenizer,
    prepare_datasets,
    ASLTranslationTrainer,
    save_model_and_tokenizer,
    calculate_model_size,
    test_model_inference
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ASL Gloss to English Translation Model')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='google/t5-efficient-tiny',
                       help='Base model name (default: google/t5-efficient-tiny)')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length (default: 128)')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='../data/gloss_data.json',
                       help='Path to training data JSON file')
    
    # Training arguments
    parser.add_argument('--output_dir', type=str, default='../models/distilt5-asl-finetuned',
                       help='Output directory for trained model')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size (default: 8)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--warmup_steps', type=int, default=500,
                       help='Number of warmup steps (default: 500)')
    
    # System arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for training')
    
    # Evaluation arguments
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Evaluation steps (default: 500)')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Model save steps (default: 500)')
    
    return parser.parse_args()

def setup_device(device_arg: str) -> str:
    """Setup and return the appropriate device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'  
        else:
            device = 'cpu'
    else:
        device = device_arg
    
    logger.info(f"Using device: {device}")
    return device

def load_training_data(data_path: str):
    """Load training data from JSON file."""
    logger.info(f"Loading training data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_data = data['train']
    val_data = data['validation'] 
    test_data = data.get('test', [])
    
    logger.info(f"Loaded {len(train_data)} training samples")
    logger.info(f"Loaded {len(val_data)} validation samples")
    logger.info(f"Loaded {len(test_data)} test samples")
    
    return train_data, val_data, test_data

def main():
    """Main training function."""
    args = parse_arguments()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save training config
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    config['device_used'] = device
    
    with open(f"{args.output_dir}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("🚀 Starting ASL Gloss to English Translation Training")
    logger.info(f"Configuration: {config}")
    
    # Load training data
    try:
        train_data, val_data, test_data = load_training_data(args.data_path)
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        logger.info("💡 Run 'python ../data/synthetic_dataset.py' to generate training data first")
        return
    
    # Load model and tokenizer
    logger.info("📥 Loading base model and tokenizer...")
    try:
        model, tokenizer = load_model_and_tokenizer(args.model_name)
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        logger.info("💡 Try using 'google/t5-efficient-tiny' or 't5-small' as model_name")
        return
    
    # Calculate model size
    model_stats = calculate_model_size(model)
    logger.info(f"📊 Model stats: {model_stats}")
    
    # Prepare datasets
    logger.info("🔄 Preparing datasets...")
    train_dataset, val_dataset = prepare_datasets(
        train_data, val_data, tokenizer, args.max_length
    )
    
    # Create trainer
    logger.info("🏋️ Setting up trainer...")
    trainer_instance = ASLTranslationTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir
    )
    
    # Training arguments for HuggingFace Trainer
    training_kwargs = {
        'num_train_epochs': args.num_epochs,
        'per_device_train_batch_size': args.batch_size,
        'per_device_eval_batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'warmup_steps': args.warmup_steps,
        'eval_steps': args.eval_steps,
        'save_steps': args.save_steps,
        'fp16': device == 'cuda',  # Use mixed precision on CUDA
        'dataloader_num_workers': 0 if device == 'mps' else 4,  # MPS doesn't support multiprocessing
    }
    
    trainer = trainer_instance.create_trainer(**training_kwargs)
    
    # Start training
    logger.info("🎯 Starting training...")
    start_time = datetime.now()
    
    try:
        # Train the model
        trainer.train()
        
        # Training completed
        end_time = datetime.now()
        training_duration = end_time - start_time
        logger.info(f"✅ Training completed in {training_duration}")
        
        # Evaluate on validation set
        logger.info("📊 Running final evaluation...")
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")
        
        # Save evaluation results
        with open(f"{args.output_dir}/eval_results.json", 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Save the final model
        logger.info("💾 Saving trained model...")
        save_model_and_tokenizer(model, tokenizer, args.output_dir)
        
        # Test inference on sample inputs
        logger.info("🧪 Testing model inference...")
        test_inputs = [
            "YESTERDAY I GO STORE WITH FRIEND",
            "MORNING COFFEE I DRINK HOT",
            "TOMORROW WORK I START EARLY", 
            "BOOK INTERESTING I READ YESTERDAY",
            "FAMILY DINNER WE EAT TOGETHER"
        ]
        
        predictions = test_model_inference(model, tokenizer, test_inputs, device)
        
        logger.info("🎭 Sample translations:")
        for inp, pred in zip(test_inputs, predictions):
            logger.info(f"  Input: {inp}")
            logger.info(f"  Output: {pred}")
            logger.info("")
        
        # Save sample results
        sample_results = {
            "test_inputs": test_inputs,
            "predictions": predictions
        }
        
        with open(f"{args.output_dir}/sample_translations.json", 'w') as f:
            json.dump(sample_results, f, indent=2)
        
        logger.info("🎉 Training pipeline completed successfully!")
        logger.info(f"📁 Model saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()