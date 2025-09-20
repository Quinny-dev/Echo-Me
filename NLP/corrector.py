"""
Enhanced Grammar Correction with Complex Sentence Handling

This version can handle complex ASL sequences and create proper English sentence structures.
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict, Optional, Tuple
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedGrammarCorrector:
    """
    Enhanced grammar correction with complex sentence restructuring.
    """
    
    def __init__(self, model_name: str = "t5-small", device: Optional[str] = None):
        """Initialize the enhanced grammar corrector."""
        self.model_name = model_name
        self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        
        # Enhanced grammar rules
        self.grammar_rules = self._init_grammar_rules()
        
        # Sentence patterns for complex restructuring
        self.sentence_patterns = self._init_sentence_patterns()
        
        # T5 generation parameters
        self.generation_params = {
            'max_length': 150,
            'num_beams': 3,
            'length_penalty': 0.9,
            'early_stopping': True,
            'do_sample': False,
            'repetition_penalty': 1.1,
        }
    
    def _load_model(self):
        """Load the T5 model and tokenizer."""
        try:
            logger.info(f"Loading T5 model: {self.model_name}")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _init_grammar_rules(self):
        """Initialize comprehensive grammar rules."""
        return {
            # Infinitive phrases
            'want go': 'want to go',
            'need go': 'need to go',
            'like go': 'like to go',
            'want eat': 'want to eat',
            'need eat': 'need to eat',
            'want buy': 'want to buy',
            'need buy': 'need to buy',
            'want see': 'want to see',
            'help move': 'help to move',
            'try eat': 'try to eat',
            'plan go': 'plan to go',
            'decide buy': 'decide to buy',
            
            # Articles with locations
            ' store': ' to the store',
            ' beach': ' to the beach',
            ' park': ' to the park',
            ' restaurant': ' to the restaurant',
            ' hospital': ' to the hospital',
            ' school': ' to the school',
            ' library': ' to the library',
            ' bank': ' to the bank',
            ' market': ' to the market',
            
            # Articles with objects
            'buy apple': 'buy an apple',
            'buy banana': 'buy a banana',
            'eat apple': 'eat an apple',
            'eat sandwich': 'eat a sandwich',
            'read book': 'read a book',
            'watch movie': 'watch a movie',
            'take photo': 'take a photo',
            'make sandwich': 'make a sandwich',
            
            # Negation
            'I not ': "I don't ",
            'You not ': "You don't ",
            'We not ': "We don't ",
            'They not ': "They don't ",
            'He not ': "He doesn't ",
            'She not ': "She doesn't ",
            ' not like': " don't like",
            ' not want': " don't want",
            ' not need': " don't need",
            ' not have': " don't have",
            
            # Question formations
            'Where you live': 'Where do you live',
            'What you want': 'What do you want',
            'How you feel': 'How do you feel',
            'Why you go': 'Why do you go',
            'When you come': 'When do you come',
            'How much cost': 'How much does it cost',
            
            # Copula (be verbs)
            'I hungry': 'I am hungry',
            'You hungry': 'You are hungry',
            'He hungry': 'He is hungry',
            'She hungry': 'She is hungry',
            'We hungry': 'We are hungry',
            'They hungry': 'They are hungry',
            'I tired': 'I am tired',
            'You tired': 'You are tired',
            'He tired': 'He is tired',
            'She tired': 'She is tired',
            'I happy': 'I am happy',
            'You happy': 'You are happy',
            'He happy': 'He is happy',
            'She happy': 'She is happy',
            'I sad': 'I am sad',
            'You sad': 'You are sad',
            
            # Past tense
            'Yesterday I eat': 'Yesterday I ate',
            'Yesterday you eat': 'Yesterday you ate',
            'Yesterday he eat': 'Yesterday he ate',
            'Yesterday she eat': 'Yesterday she ate',
            'Yesterday we eat': 'Yesterday we ate',
            'Yesterday they eat': 'Yesterday they ate',
            'Yesterday I go': 'Yesterday I went',
            'Yesterday you go': 'Yesterday you went',
            'Yesterday he go': 'Yesterday he went',
            'Yesterday she go': 'Yesterday she went',
            
            # Future tense
            'Tomorrow I go': 'Tomorrow I will go',
            'Tomorrow you go': 'Tomorrow you will go',
            'Tomorrow he go': 'Tomorrow he will go',
            'Tomorrow she go': 'Tomorrow she will go',
            'Tomorrow we go': 'Tomorrow we will go',
            'Tomorrow they go': 'Tomorrow they will go',
            
            # Pronouns
            'help I': 'help me',
            'give I': 'give me',
            'tell I': 'tell me',
            'show I': 'show me',
            'teach I': 'teach me',
        }
    
    def _init_sentence_patterns(self):
        """Initialize patterns for complex sentence restructuring."""
        return {
            'vacation_activities': {
                'keywords': ['holiday', 'trip', 'vacation', 'beach', 'swim', 'photo', 'sunset', 'lunch'],
                'template': "On our {location} {trip_type}, we {activities}."
            },
            'daily_activities': {
                'keywords': ['morning', 'afternoon', 'work', 'lunch', 'home', 'dinner'],
                'template': "During the day, I {activities}."
            },
            'shopping': {
                'keywords': ['store', 'buy', 'market', 'shop', 'money', 'cost'],
                'template': "I went to {location} to {action} {items}."
            },
            'food_activities': {
                'keywords': ['eat', 'lunch', 'dinner', 'breakfast', 'cook', 'restaurant'],
                'template': "For {meal}, I {action} {food}."
            }
        }
    
    def _identify_sentence_type(self, words: List[str]) -> str:
        """Identify the type of sentence to help with restructuring."""
        word_set = set(word.lower() for word in words)
        
        for pattern_type, pattern_info in self.sentence_patterns.items():
            keyword_matches = sum(1 for keyword in pattern_info['keywords'] if keyword in word_set)
            if keyword_matches >= 2:  # Need at least 2 keyword matches
                return pattern_type
        
        return 'simple'
    
    def _restructure_complex_sentence(self, text: str) -> str:
        """Restructure complex sentences with multiple activities/items."""
        words = text.strip(' .').split()
        
        if len(words) < 5:  # Simple sentences don't need restructuring
            return text
        
        sentence_type = self._identify_sentence_type(words)
        
        if sentence_type == 'vacation_activities':
            return self._restructure_vacation_sentence(words)
        elif sentence_type == 'shopping':
            return self._restructure_shopping_sentence(words)
        elif sentence_type == 'daily_activities':
            return self._restructure_daily_sentence(words)
        else:
            return self._restructure_list_sentence(words)
    
    def _restructure_vacation_sentence(self, words: List[str]) -> str:
        """Restructure vacation-related sentences."""
        # Common vacation words
        locations = ['beach', 'park', 'resort', 'hotel']
        activities = ['swim', 'walk', 'run', 'play', 'eat', 'drink', 'photo', 'lunch', 'dinner']
        descriptors = ['beautiful', 'hot', 'cold', 'sunny', 'fun', 'relaxing']
        
        # Find location
        location = None
        for word in words:
            if word.lower() in locations:
                location = word.lower()
                break
        
        # Find activities
        found_activities = []
        for word in words:
            if word.lower() in activities:
                found_activities.append(word.lower())
        
        # Build sentence
        if location and found_activities:
            if len(found_activities) == 1:
                return f"We went to the {location} and {found_activities[0]}."
            elif len(found_activities) == 2:
                return f"We went to the {location} and {found_activities[0]} and {found_activities[1]}."
            else:
                activity_list = ', '.join(found_activities[:-1]) + f", and {found_activities[-1]}"
                return f"We went to the {location} and {activity_list}."
        
        # Fallback to list restructuring
        return self._restructure_list_sentence(words)
    
    def _restructure_shopping_sentence(self, words: List[str]) -> str:
        """Restructure shopping-related sentences."""
        # Look for shopping patterns
        if 'store' in [w.lower() for w in words] or 'shop' in [w.lower() for w in words]:
            items = []
            for word in words:
                if word.lower() in ['apple', 'banana', 'bread', 'milk', 'food', 'coffee', 'tea']:
                    items.append(word.lower())
            
            if items:
                if len(items) == 1:
                    return f"I went to the store to buy {items[0]}."
                else:
                    item_list = ', '.join(items[:-1]) + f", and {items[-1]}"
                    return f"I went to the store to buy {item_list}."
        
        return self._restructure_list_sentence(words)
    
    def _restructure_daily_sentence(self, words: List[str]) -> str:
        """Restructure daily activity sentences."""
        # Simple daily activity restructuring
        return self._restructure_list_sentence(words)
    
    def _restructure_list_sentence(self, words: List[str]) -> str:
        """Restructure sentences with lists of items/activities."""
        if len(words) <= 4:
            return ' '.join(words)
        
        # Group words by type
        subjects = []
        verbs = []
        objects = []
        others = []
        
        verb_words = ['go', 'eat', 'drink', 'buy', 'see', 'watch', 'play', 'swim', 'walk', 'run', 'take', 'make', 'apply']
        subject_words = ['I', 'you', 'he', 'she', 'we', 'they', 'family', 'friend', 'friends']
        
        for word in words:
            word_lower = word.lower()
            if word_lower in subject_words or word in ['I']:
                subjects.append(word)
            elif word_lower in verb_words:
                verbs.append(word_lower)
            elif len(word) > 2:  # Potential objects
                objects.append(word_lower)
            else:
                others.append(word)
        
        # Build a more natural sentence
        if subjects and verbs:
            subject = subjects[0] if subjects else "We"
            
            if len(verbs) == 1 and len(objects) >= 3:
                # Single verb, multiple objects: "We did X, Y, and Z"
                verb = verbs[0]
                if len(objects) <= 2:
                    object_phrase = ' and '.join(objects)
                else:
                    object_phrase = ', '.join(objects[:-1]) + f', and {objects[-1]}'
                return f"{subject} {verb} {object_phrase}."
            
            elif len(verbs) >= 2:
                # Multiple verbs: "We did X, did Y, and did Z"
                if len(verbs) <= 2:
                    verb_phrase = ' and '.join(verbs)
                else:
                    verb_phrase = ', '.join(verbs[:-1]) + f', and {verbs[-1]}'
                return f"{subject} {verb_phrase}."
        
        # Fallback: just join with commas and conjunctions
        if len(words) <= 3:
            return ' '.join(words)
        else:
            return ', '.join(words[:-1]) + f', and {words[-1]}'
    
    def _apply_comprehensive_rules(self, text: str) -> str:
        """Apply all grammar rules comprehensively."""
        corrected = text
        
        # First, apply basic grammar rules
        for pattern, replacement in self.grammar_rules.items():
            corrected = corrected.replace(pattern, replacement)
        
        # Then, handle complex sentence restructuring
        corrected = self._restructure_complex_sentence(corrected)
        
        # Additional cleanup
        corrected = re.sub(r'\b(a|an|the)\s+(a|an|the)\s+', r'\1 ', corrected)  # Remove double articles
        corrected = re.sub(r'\s+([.!?])', r'\1', corrected)  # Fix punctuation spacing
        corrected = re.sub(r'\s+,', ',', corrected)  # Fix comma spacing
        corrected = re.sub(r'\s+', ' ', corrected)  # Normalize spaces
        
        return corrected.strip()
    
    def correct_single(self, rough_text: str) -> Dict[str, any]:
        """
        Correct a single sentence with enhanced processing.
        """
        if not rough_text or not rough_text.strip():
            return {
                'input': rough_text,
                'corrected': '',
                'confidence_score': 0.0,
                'model_used': self.model_name,
                'method': 'none'
            }
        
        original_text = rough_text.strip()
        
        # Method 1: Enhanced rule-based correction
        rule_output = self._apply_comprehensive_rules(original_text)
        
        # Method 2: Try T5 with better prompting
        try:
            t5_input = f"improve this sentence: {original_text}"
            input_ids = self.tokenizer.encode(t5_input, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(input_ids=input_ids, **self.generation_params)
            
            t5_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            t5_output = re.sub(r'^(improve this sentence:|grammar:|correct:|fix:)\s*', '', t5_output, flags=re.IGNORECASE)
            
            # If T5 output is reasonable, use it
            if (len(t5_output) > len(original_text) * 0.5 and 
                len(t5_output) < len(original_text) * 2 and
                not t5_output.lower().startswith(('improve', 'grammar', 'correct'))):
                t5_output = self._apply_comprehensive_rules(t5_output)
            else:
                t5_output = rule_output  # Fallback to rule-based
                
        except Exception as e:
            logger.warning(f"T5 processing failed: {e}")
            t5_output = rule_output
        
        # Choose the better output
        rule_score = self._score_output(original_text, rule_output)
        t5_score = self._score_output(original_text, t5_output)
        
        if t5_score > rule_score and t5_score > 0.3:
            final_output = t5_output
            method = 'hybrid-t5'
            confidence = t5_score
        else:
            final_output = rule_output
            method = 'rule-enhanced'
            confidence = rule_score
        
        # Final post-processing
        final_output = self._post_process(final_output)
        
        return {
            'input': original_text,
            'corrected': final_output,
            'confidence_score': confidence,
            'model_used': self.model_name,
            'method': method
        }
    
    def _score_output(self, original: str, corrected: str) -> float:
        """Score the quality of correction."""
        if not corrected or len(corrected) < 3:
            return 0.0
        
        orig_words = original.lower().split()
        corr_words = corrected.lower().split()
        
        # Content preservation
        content_words = set(orig_words) - {'the', 'a', 'an', 'to', 'and', 'or', 'but', 'in', 'on', 'at'}
        preserved = sum(1 for word in content_words if word in corr_words)
        preservation_score = preserved / max(len(content_words), 1)
        
        # Grammar improvement indicators
        grammar_indicators = {'to', 'the', 'a', 'an', 'and', 'or', 'but', 'do', 'does', 'did', 'will', 'would', 'can', 'could'}
        grammar_count = sum(1 for word in corr_words if word in grammar_indicators)
        grammar_score = min(grammar_count / len(corr_words), 0.4)
        
        # Length reasonableness
        length_ratio = min(len(corr_words), len(orig_words)) / max(len(corr_words), len(orig_words))
        
        # Sentence structure (periods, commas)
        structure_score = 0.1 if ('.' in corrected or '!' in corrected or '?' in corrected) else 0
        
        total_score = (preservation_score * 0.4) + (grammar_score * 0.3) + (length_ratio * 0.2) + structure_score
        return min(total_score, 1.0)
    
    def _post_process(self, text: str) -> str:
        """Final post-processing."""
        if not text:
            return text
        
        text = text.strip()
        
        # Capitalize first letter
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        # Ensure proper ending punctuation
        if text and not text[-1] in '.!?':
            text += '.'
        
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.!?,])', r'\1', text)
        
        return text
    
    def correct_batch(self, rough_texts: List[str]) -> List[Dict[str, any]]:
        """Process multiple sentences."""
        return [self.correct_single(text) for text in rough_texts]
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024),
            'correction_methods': ['rule-enhanced', 'hybrid-t5', 'sentence-restructuring']
        }


# For backward compatibility, create an alias
GrammarCorrector = EnhancedGrammarCorrector

# Example usage and testing
if __name__ == "__main__":
    corrector = EnhancedGrammarCorrector()
    
    test_sentences = [
        "I want go store buy apple.",
        "Holiday trip beach family swim sandcastle sunburn apply lotion lunch seafood icecream sunset photo.",
        "Yesterday you eat pizza restaurant.",
        "Where you live.",
        "I not like cold weather.",
        "Can you help I please.",
        "Tomorrow we meet friend coffee shop.",
        "Me hungry want eat sandwich now."
    ]
    
    print("Enhanced Grammar Corrector Test")
    print("=" * 70)
    
    for sentence in test_sentences:
        result = corrector.correct_single(sentence)
        print(f"Input:  {result['input']}")
        print(f"Output: {result['corrected']}")
        print(f"Method: {result['method']} (confidence: {result['confidence_score']:.2f})")
        print("-" * 50)