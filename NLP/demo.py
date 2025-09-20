"""
ASL-to-English Translation Pipeline Demo

This script demonstrates the complete pipeline:
1. ASL gloss input
2. Rule-based translation to rough English
3. T5-based grammar correction to fluent English

Run this script to see the full system in action.
"""

import sys
import time
from typing import Dict, List
import json

# Import our modules
from rule_based import RuleBasedTranslator
from corrector import GrammarCorrector


class ASLTranslationPipeline:
    """
    Complete ASL-to-English translation pipeline combining rule-based
    translation with neural grammar correction.
    """
    
    def __init__(self, model_name: str = "t5-small"):
        """
        Initialize the translation pipeline.
        
        Args:
            model_name: Name of the T5 model for grammar correction
        """
        print("Initializing ASL Translation Pipeline...")
        print("-" * 50)
        
        # Initialize components
        print("Loading rule-based translator...")
        self.rule_translator = RuleBasedTranslator()
        
        print("Loading T5 grammar corrector...")
        self.grammar_corrector = GrammarCorrector(model_name=model_name)
        
        print("Pipeline ready!")
        print("=" * 50)
    
    def translate(self, asl_gloss: str, show_steps: bool = True) -> Dict[str, any]:
        """
        Translate ASL gloss to fluent English.
        
        Args:
            asl_gloss: Input ASL gloss string
            show_steps: Whether to display intermediate steps
            
        Returns:
            Dictionary with complete translation results
        """
        if show_steps:
            print(f"Input ASL Gloss: {asl_gloss}")
            print("-" * 30)
        
        # Step 1: Rule-based translation
        rule_result = self.rule_translator.translate(asl_gloss)
        rough_english = rule_result['rough_english']
        
        if show_steps:
            print(f"Step 1 - Rule-based translation:")
            print(f"  Rough English: {rough_english}")
            print(f"  Detected tense: {rule_result['detected_tense']}")
            print()
        
        # Step 2: Grammar correction
        correction_result = self.grammar_corrector.correct_single(rough_english)
        fluent_english = correction_result['corrected']
        
        if show_steps:
            print(f"Step 2 - Grammar correction:")
            print(f"  Fluent English: {fluent_english}")
            print(f"  Confidence: {correction_result['confidence_score']:.2f}")
            print()
        
        # Combine results
        final_result = {
            'input_gloss': asl_gloss,
            'rule_translation': rule_result,
            'grammar_correction': correction_result,
            'final_output': fluent_english,
            'processing_steps': {
                'cleaned_gloss': rule_result['cleaned_gloss'],
                'detected_signs': rule_result['signs'],
                'detected_tense': rule_result['detected_tense'],
                'rough_english': rough_english,
                'fluent_english': fluent_english,
                'confidence_score': correction_result['confidence_score']
            }
        }
        
        return final_result
    
    def translate_batch(self, asl_glosses: List[str], show_progress: bool = True) -> List[Dict[str, any]]:
        """
        Translate multiple ASL glosses efficiently.
        
        Args:
            asl_glosses: List of ASL gloss strings
            show_progress: Whether to show progress updates
            
        Returns:
            List of translation results
        """
        results = []
        
        if show_progress:
            print(f"Processing {len(asl_glosses)} ASL glosses...")
            print("-" * 40)
        
        # Step 1: Rule-based translations for all inputs
        rule_results = []
        for i, gloss in enumerate(asl_glosses):
            if show_progress:
                print(f"Rule-based translation {i+1}/{len(asl_glosses)}: {gloss}")
            
            rule_result = self.rule_translator.translate(gloss)
            rule_results.append(rule_result)
        
        # Step 2: Batch grammar correction
        rough_sentences = [result['rough_english'] for result in rule_results]
        
        if show_progress:
            print("\nApplying grammar corrections...")
        
        correction_results = self.grammar_corrector.correct_batch(rough_sentences)
        
        # Combine results
        for rule_result, correction_result in zip(rule_results, correction_results):
            final_result = {
                'input_gloss': rule_result['original_gloss'],
                'rule_translation': rule_result,
                'grammar_correction': correction_result,
                'final_output': correction_result['corrected'],
                'processing_steps': {
                    'cleaned_gloss': rule_result['cleaned_gloss'],
                    'detected_signs': rule_result['signs'],
                    'detected_tense': rule_result['detected_tense'],
                    'rough_english': rule_result['rough_english'],
                    'fluent_english': correction_result['corrected'],
                    'confidence_score': correction_result['confidence_score']
                }
            }
            results.append(final_result)
        
        if show_progress:
            print(f"Completed processing {len(results)} translations!")
            print("=" * 50)
        
        return results
    
    def get_system_info(self) -> Dict[str, any]:
        """Get information about the translation system."""
        return {
            'rule_translator': {
                'dictionary_size': len(self.rule_translator.sign_dictionary),
                'temporal_markers': len(self.rule_translator.temporal_markers)
            },
            'grammar_corrector': self.grammar_corrector.get_model_info()
        }


def run_demo():
    """Run the main demo with predefined examples."""
    
    # Initialize pipeline
    pipeline = ASLTranslationPipeline()
    
    # Demo examples covering various ASL structures
    demo_examples = [
        "ME WANT GO STORE BUY APPLE",
        "YESTERDAY YOU EAT PIZZA RESTAURANT",
        "WHERE YOU LIVE QUESTION",
        "ME NOT LIKE COLD WEATHER WINTER",
        "TOMORROW WE MEET FRIEND COFFEE SHOP",
        "BOOK RED VERY INTERESTING READ",
        "HOW MUCH COST NEW CAR",
        "ME FINISH HOMEWORK ALREADY DONE",
        "CAN YOU HELP ME MOVE HOUSE",
        "COFFEE HOT BUT TASTE GOOD",
        "ME HUNGRY WANT EAT SANDWICH",
        "SHE BEAUTIFUL WOMAN SMART TOO",
        "DOG BROWN RUN FAST PARK",
        "WILL RAIN TOMORROW MAYBE",
        "ME STUDY UNIVERSITY FOUR YEAR"
    ]
    
    print("ASL-to-English Translation Pipeline Demo")
    print("=" * 60)
    print("This demo shows the complete translation process:")
    print("1. ASL Gloss → Rule-based → Rough English")
    print("2. Rough English → T5 Grammar Correction → Fluent English")
    print("=" * 60)
    print()
    
    # Run individual examples with detailed steps
    print("INDIVIDUAL TRANSLATIONS (with detailed steps):")
    print("=" * 60)
    
    for i, gloss in enumerate(demo_examples[:5], 1):  # Show first 5 with details
        print(f"\nExample {i}:")
        result = pipeline.translate(gloss, show_steps=True)
        print("=" * 40)
    
    # Run batch processing for remaining examples
    print("\nBATCH PROCESSING (remaining examples):")
    print("=" * 60)
    
    remaining_examples = demo_examples[5:]
    batch_results = pipeline.translate_batch(remaining_examples, show_progress=True)
    
    print("\nBatch Results Summary:")
    print("-" * 40)
    for i, result in enumerate(batch_results, 6):
        print(f"{i:2d}. {result['input_gloss']}")
        print(f"    → {result['final_output']}")
        print(f"    (Confidence: {result['processing_steps']['confidence_score']:.2f})")
        print()
    
    # Show system information
    print("SYSTEM INFORMATION:")
    print("-" * 40)
    sys_info = pipeline.get_system_info()
    print(f"Rule-based dictionary size: {sys_info['rule_translator']['dictionary_size']} signs")
    print(f"Grammar model: {sys_info['grammar_corrector']['model_name']}")
    print(f"Model parameters: {sys_info['grammar_corrector']['num_parameters']:,}")
    print(f"Model size: {sys_info['grammar_corrector']['model_size_mb']:.1f} MB")
    print(f"Device: {sys_info['grammar_corrector']['device']}")
    
    return pipeline


def interactive_mode(pipeline: ASLTranslationPipeline):
    """Run interactive mode where user can input their own ASL glosses."""
    
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Enter ASL glosses to see the translation process.")
    print("Type 'quit', 'exit', or 'q' to stop.")
    print("Type 'batch' to enter multiple glosses at once.")
    print("Type 'help' for more options.")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nASL Gloss: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Enter any ASL gloss for translation")
                print("  - 'batch' - Enter multiple glosses")
                print("  - 'quit'/'exit'/'q' - Exit the program")
                print("  - 'help' - Show this help")
                continue
            
            elif user_input.lower() == 'batch':
                print("\nBatch mode - enter multiple ASL glosses (one per line).")
                print("Type 'END' on a new line when finished:")
                
                batch_glosses = []
                while True:
                    batch_input = input("  ").strip()
                    if batch_input.upper() == 'END':
                        break
                    if batch_input:
                        batch_glosses.append(batch_input)
                
                if batch_glosses:
                    print(f"\nProcessing {len(batch_glosses)} glosses...")
                    batch_results = pipeline.translate_batch(batch_glosses)
                    
                    print("\nBatch Results:")
                    print("-" * 40)
                    for i, result in enumerate(batch_results, 1):
                        print(f"{i}. {result['input_gloss']}")
                        print(f"   → {result['final_output']}")
                        print()
                else:
                    print("No glosses entered.")
                continue
            
            elif not user_input:
                continue
            
            else:
                # Regular translation
                result = pipeline.translate(user_input, show_steps=True)
                print("✓ Translation completed!")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or type 'help' for options.")


def main():
    """Main function to run the demo."""
    
    print("Starting ASL Translation Pipeline...")
    print("This may take a moment to load the T5 model...")
    print()
    
    try:
        # Run the demo
        pipeline = run_demo()
        
        # Ask if user wants interactive mode
        while True:
            try:
                response = input("\nWould you like to try interactive mode? (y/n): ").strip().lower()
                if response in ['y', 'yes']:
                    interactive_mode(pipeline)
                    break
                elif response in ['n', 'no']:
                    print("Demo completed. Thank you!")
                    break
                else:
                    print("Please enter 'y' or 'n'")
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
    
    except Exception as e:
        print(f"Error running demo: {e}")
        print("Please check that all dependencies are installed correctly.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)