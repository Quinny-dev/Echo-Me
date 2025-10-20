"""
Inference script for ASL Gloss to English translation.
Loads trained model and provides translation functionality.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Union

import torch

from model_utils import load_trained_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASLTranslator:
    """ASL Gloss to English translator class."""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize translator.
        
        Args:
            model_path: Path to trained model
            device: Device to run inference on ('auto', 'cpu', 'cuda', 'mps')
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        
        logger.info(f"Loading model from {model_path}")
        self.model, self.tokenizer = load_trained_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _setup_device(self, device_arg: str) -> str:
        """Setup and return appropriate device."""
        if device_arg == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device_arg
    
    def translate(self, gloss_input: str, max_length: int = 128, 
                 num_beams: int = 4, do_sample: bool = False) -> str:
        """
        Translate ASL gloss to English.
        
        Args:
            gloss_input: Input gloss text
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            
        Returns:
            English translation
        """
        # Clean and prepare input
        gloss_input = gloss_input.strip()
        if not gloss_input:
            return ""
        
        # Add task prefix
        input_text = f"translate ASL gloss to English: {gloss_input}"
        
        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=max_length,
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Generate translation
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=do_sample,
                temperature=0.7 if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode output
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        return translation.strip()
    
    def translate_batch(self, gloss_inputs: List[str], max_length: int = 128,
                       num_beams: int = 4, batch_size: int = 8) -> List[str]:
        """
        Translate multiple gloss inputs in batches.
        
        Args:
            gloss_inputs: List of gloss texts
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            batch_size: Batch size for processing
            
        Returns:
            List of English translations
        """
        translations = []
        
        # Process in batches
        for i in range(0, len(gloss_inputs), batch_size):
            batch = gloss_inputs[i:i + batch_size]
            
            # Prepare batch inputs
            input_texts = [f"translate ASL gloss to English: {gloss.strip()}" 
                          for gloss in batch]
            
            with torch.no_grad():
                # Tokenize batch
                inputs = self.tokenizer(
                    input_texts,
                    return_tensors="pt",
                    max_length=max_length,
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # Generate translations
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode outputs
                batch_translations = self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
                
                translations.extend([t.strip() for t in batch_translations])
        
        return translations
    
    def translate_paragraph(self, paragraph_gloss: str, sentence_separator: str = ". ") -> str:
        """
        Translate paragraph-level gloss input.
        
        Args:
            paragraph_gloss: Multi-sentence gloss input
            sentence_separator: Separator for sentences in output
            
        Returns:
            English paragraph translation
        """
        # Split into sentences (simple heuristic)
        sentences = []
        current_sentence = []
        
        words = paragraph_gloss.strip().split()
        for word in words:
            current_sentence.append(word)
            # Simple sentence boundary detection
            if word.endswith('.') or len(current_sentence) > 15:
                if current_sentence:
                    sentences.append(' '.join(current_sentence).rstrip('.'))
                    current_sentence = []
        
        # Add remaining words as a sentence
        if current_sentence:
            sentences.append(' '.join(current_sentence))
        
        if not sentences:
            return self.translate(paragraph_gloss)
        
        # Translate each sentence
        translations = []
        for sentence in sentences:
            if sentence.strip():
                translation = self.translate(sentence.strip())
                translations.append(translation)
        
        # Join translations
        return sentence_separator.join(translations)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ASL Gloss to English Translation Inference')
    
    parser.add_argument('--model_path', type=str, 
                       default='../models/distilt5-asl-finetuned',
                       help='Path to trained model directory')
    parser.add_argument('--input', type=str,
                       help='Input gloss text to translate')
    parser.add_argument('--input_file', type=str,
                       help='Path to file containing gloss inputs (one per line)')
    parser.add_argument('--output_file', type=str,
                       help='Path to save translations (optional)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for inference')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum generation length')
    parser.add_argument('--num_beams', type=int, default=4,
                       help='Number of beams for beam search')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive translation mode')
    parser.add_argument('--paragraph_mode', action='store_true',
                       help='Treat input as paragraph (multiple sentences)')
    
    return parser.parse_args()

def interactive_mode(translator: ASLTranslator):
    """Interactive translation mode."""
    print("\n🎯 Interactive ASL Gloss Translation Mode")
    print("Enter ASL gloss text (type 'quit' to exit, 'help' for examples)")
    print("-" * 50)
    
    while True:
        try:
            gloss_input = input("\nGloss: ").strip()
            
            if gloss_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif gloss_input.lower() == 'help':
                print_examples()
                continue
            elif not gloss_input:
                continue
            
            # Translate
            translation = translator.translate(gloss_input)
            print(f"English: {translation}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def print_examples():
    """Print example gloss inputs."""
    examples = [
        "YESTERDAY I GO STORE WITH FRIEND",
        "MORNING COFFEE I DRINK HOT",
        "BOOK INTERESTING I READ LAST WEEK",
        "TOMORROW WORK I START EARLY",
        "FAMILY DINNER WE EAT TOGETHER",
        "WEATHER TODAY BEAUTIFUL SUNNY",
        "MOVIE LAST NIGHT WE WATCH FUNNY",
        "SCHOOL I STUDY MATH DIFFICULT",
        # Paragraph example
        "YESTERDAY MORNING I WAKE-UP EARLY. COFFEE I DRINK FIRST. WORK I GO BY CAR. MEETING IMPORTANT I HAVE."
    ]
    
    print("\n📝 Example inputs:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    print()

def load_input_file(file_path: str) -> List[str]:
    """Load gloss inputs from file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Filter out empty lines
    return [line for line in lines if line]

def save_translations(inputs: List[str], translations: List[str], output_file: str):
    """Save translations to file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ASL Gloss to English Translations\n")
        f.write("=" * 50 + "\n\n")
        
        for i, (inp, trans) in enumerate(zip(inputs, translations), 1):
            f.write(f"{i}. Gloss: {inp}\n")
            f.write(f"   English: {trans}\n\n")

def main():
    """Main inference function."""
    args = parse_arguments()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"❌ Model not found: {args.model_path}")
        logger.info("💡 Train the model first using: python train.py")
        return
    
    # Initialize translator
    try:
        translator = ASLTranslator(args.model_path, args.device)
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return
    
    # Interactive mode
    if args.interactive:
        interactive_mode(translator)
        return
    
    # Process single input
    if args.input:
        logger.info(f"Input: {args.input}")
        
        if args.paragraph_mode:
            translation = translator.translate_paragraph(args.input)
        else:
            translation = translator.translate(args.input)
        
        print(f"Translation: {translation}")
        
        # Save to file if specified
        if args.output_file:
            save_translations([args.input], [translation], args.output_file)
            logger.info(f"Translation saved to {args.output_file}")
        
        return
    
    # Process input file
    if args.input_file:
        try:
            inputs = load_input_file(args.input_file)
            logger.info(f"Loaded {len(inputs)} inputs from {args.input_file}")
            
            # Translate all inputs
            logger.info("🔄 Translating...")
            if args.paragraph_mode:
                translations = [translator.translate_paragraph(inp) for inp in inputs]
            else:
                translations = translator.translate_batch(inputs)
            
            # Display results
            for i, (inp, trans) in enumerate(zip(inputs, translations), 1):
                print(f"\n{i}. Gloss: {inp}")
                print(f"   English: {trans}")
            
            # Save results if specified
            if args.output_file:
                save_translations(inputs, translations, args.output_file)
                logger.info(f"Translations saved to {args.output_file}")
            
        except FileNotFoundError as e:
            logger.error(f"❌ {e}")
        except Exception as e:
            logger.error(f"❌ Error processing file: {e}")
        
        return
    
    # No input provided - show help
    print("❌ No input provided!")
    print("\nUsage examples:")
    print('  python inference.py --input "YESTERDAY I GO STORE"')
    print('  python inference.py --input_file sample_inputs.txt')
    print('  python inference.py --interactive')
    print('\nFor more options: python inference.py --help')

if __name__ == "__main__":
    main()