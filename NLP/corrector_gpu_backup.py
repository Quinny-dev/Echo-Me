"""
Grammar Correction Model using T5-small

This module takes rough English from the rule-based translator and converts it
to fluent, grammatically correct English using a pre-trained T5 model.
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GrammarCorrector:
    """
    Grammar correction using T5-small model from HuggingFace.
    
    This class loads a pre-trained T5 model and uses it to convert rough English
    sentences into grammatically correct, fluent English.
    """
    
    def __init__(self, model_name: str = "t5-small", device: Optional[str] = None):
        """
        Initialize the grammar corrector.
        
        Args:
            model_name: Name of the T5 model to use (default: "t5-small")
            device: Device to run model on. If None, auto-detects GPU/CPU
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        
        # Generation parameters for better output
        self.generation_params = {
            'max_length': 128,
            'num_beams': 4,
            'length_penalty': 1.0,
            'early_stopping': True,
            'do_sample': False,  # Use beam search for more consistent results
            'temperature': 1.0,
            'repetition_penalty': 1.1,
        }
    
    def _load_model(self):
        """Load the T5 model and tokenizer."""
        try:
            logger.info(f"Loading T5 model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _prepare_input(self, rough_text: str) -> str:
        """
        Prepare input text for T5 model.
        
        T5 expects a task prefix. We'll use "grammar: " to indicate
        grammar correction task.
        
        Args:
            rough_text: Rough English text to correct
            
        Returns:
            Formatted input for T5
        """
        # Remove any existing punctuation and normalize
        text = rough_text.strip()
        if text.endswith('.'):
            text = text[:-1]
        
        # Add task prefix for T5
        return f"grammar: {text}"
    
    def correct_single(self, rough_text: str) -> Dict[str, any]:
        """
        Correct a single rough English sentence.
        
        Args:
            rough_text: Rough English sentence to correct
            
        Returns:
            Dictionary with correction results and metadata
        """
        if not rough_text or not rough_text.strip():
            return {
                'input': rough_text,
                'corrected': '',
                'confidence_score': 0.0,
                'model_used': self.model_name
            }
        
        try:
            # Prepare input
            input_text = self._prepare_input(rough_text)
            
            # Tokenize input
            input_ids = self.tokenizer.encode(
                input_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            ).to(self.device)
            
            # Generate correction
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    **self.generation_params
                )
            
            # Decode output
            corrected_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            ).strip()
            
            # Post-process the output
            corrected_text = self._post_process(corrected_text)
            
            # Calculate a simple confidence score based on length difference
            # (This is a heuristic - in a real system you might want more sophisticated scoring)
            confidence_score = self._calculate_confidence(rough_text, corrected_text)
            
            return {
                'input': rough_text,
                'corrected': corrected_text,
                'confidence_score': confidence_score,
                'model_used': self.model_name,
                'input_length': len(rough_text.split()),
                'output_length': len(corrected_text.split())
            }
            
        except Exception as e:
            logger.error(f"Error during correction: {e}")
            return {
                'input': rough_text,
                'corrected': rough_text,  # Fallback to original
                'confidence_score': 0.0,
                'model_used': self.model_name,
                'error': str(e)
            }
    
    def correct_batch(self, rough_texts: List[str]) -> List[Dict[str, any]]:
        """
        Correct multiple sentences in batch for efficiency.
        
        Args:
            rough_texts: List of rough English sentences
            
        Returns:
            List of correction results
        """
        results = []
        
        # Process in batches to avoid memory issues
        batch_size = 4  # Adjust based on your GPU memory
        
        for i in range(0, len(rough_texts), batch_size):
            batch = rough_texts[i:i + batch_size]
            
            try:
                # Prepare inputs
                input_texts = [self._prepare_input(text) for text in batch]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    input_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Generate corrections
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        **self.generation_params
                    )
                
                # Decode outputs
                corrected_batch = self.tokenizer.batch_decode(
                    outputs, 
                    skip_special_tokens=True
                )
                
                # Process results
                for orig, corrected in zip(batch, corrected_batch):
                    corrected = self._post_process(corrected)
                    confidence = self._calculate_confidence(orig, corrected)
                    
                    results.append({
                        'input': orig,
                        'corrected': corrected,
                        'confidence_score': confidence,
                        'model_used': self.model_name
                    })
                    
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                # Add fallback results for this batch
                for text in batch:
                    results.append({
                        'input': text,
                        'corrected': text,
                        'confidence_score': 0.0,
                        'model_used': self.model_name,
                        'error': str(e)
                    })
        
        return results
    
    def _post_process(self, text: str) -> str:
        """
        Post-process the T5 output to clean it up.
        
        Args:
            text: Raw T5 output
            
        Returns:
            Cleaned text
        """
        # Basic cleaning
        text = text.strip()
        
        # Ensure proper capitalization
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        # Ensure proper punctuation
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Fix common issues
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        text = text.replace(' !', '!')
        text = text.replace(' ?', '?')
        
        # Remove duplicate spaces
        text = ' '.join(text.split())
        
        return text
    
    def _calculate_confidence(self, input_text: str, output_text: str) -> float:
        """
        Calculate a simple confidence score.
        
        This is a heuristic based on the assumption that good corrections
        should be similar in length but more grammatical.
        
        Args:
            input_text: Original rough text
            output_text: Corrected text
            
        Returns:
            Confidence score between 0 and 1
        """
        if not input_text or not output_text:
            return 0.0
        
        # Length similarity (corrections shouldn't be drastically different)
        input_len = len(input_text.split())
        output_len = len(output_text.split())
        
        if input_len == 0:
            return 0.0
        
        length_ratio = min(input_len, output_len) / max(input_len, output_len)
        
        # Word overlap (good corrections should preserve most content words)
        input_words = set(input_text.lower().split())
        output_words = set(output_text.lower().split())
        
        if len(input_words) == 0:
            return 0.0
        
        word_overlap = len(input_words.intersection(output_words)) / len(input_words)
        
        # Combine metrics (you can adjust weights)
        confidence = (length_ratio * 0.4) + (word_overlap * 0.6)
        
        return min(confidence, 1.0)
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the grammar corrector
    corrector = GrammarCorrector()
    
    # Test sentences (rough English that needs correction)
    test_sentences = [
        "I want go store buy apple.",
        "Yesterday you eat pizza.",
        "Where you live.",
        "I not like cold weather.",
        "Tomorrow we meet friend restaurant.",
        "Book red very interesting.",
        "How much cost car.",
        "I finish homework already.",
        "Can you help me please.",
        "Coffee hot but good."
    ]
    
    print("Grammar Corrector Test")
    print("=" * 50)
    
    # Test single corrections
    print("\nSingle Sentence Corrections:")
    print("-" * 30)
    
    for sentence in test_sentences:
        result = corrector.correct_single(sentence)
        print(f"Input: {result['input']}")
        print(f"Corrected: {result['corrected']}")
        print(f"Confidence: {result['confidence_score']:.2f}")
        print("-" * 30)
    
    # Test batch processing
    print("\nBatch Processing Test:")
    print("-" * 30)
    
    batch_results = corrector.correct_batch(test_sentences[:3])
    for result in batch_results:
        print(f"Input: {result['input']}")
        print(f"Corrected: {result['corrected']}")
        print(f"Confidence: {result['confidence_score']:.2f}")
        print("-" * 20)
    
    # Show model info
    print("\nModel Information:")
    print("-" * 30)
    model_info = corrector.get_model_info()
    for key, value in model_info.items():
        print(f"{key}: {value}")