"""
Rule-based ASL Gloss to Rough English Translator

This module converts ASL gloss (e.g., "ME WANT GO STORE") to rough English
(e.g., "I want go store") using dictionary mapping and basic grammar rules.
"""

import re
from typing import List, Dict, Tuple, Optional


class RuleBasedTranslator:
    """
    Converts ASL gloss to rough English using dictionary mapping and basic rules.
    
    The output is intentionally "rough" - it will be refined by the grammar correction model.
    Focus is on getting the basic meaning across with correct vocabulary.
    """
    
    def __init__(self):
        # Core sign-to-word dictionary
        self.sign_dictionary = {
            # Pronouns
            'ME': 'I', 'MY': 'my', 'MINE': 'mine',
            'YOU': 'you', 'YOUR': 'your', 'YOURS': 'yours',
            'HE': 'he', 'SHE': 'she', 'IT': 'it',
            'WE': 'we', 'THEY': 'they', 'US': 'us', 'THEM': 'them',
            'HIS': 'his', 'HER': 'her', 'THEIR': 'their',
            
            # Question words
            'WHO': 'who', 'WHAT': 'what', 'WHERE': 'where',
            'WHEN': 'when', 'WHY': 'why', 'HOW': 'how', 'WHICH': 'which',
            
            # Common verbs
            'WANT': 'want', 'NEED': 'need', 'LIKE': 'like', 'LOVE': 'love',
            'GO': 'go', 'COME': 'come', 'LEAVE': 'leave', 'ARRIVE': 'arrive',
            'SEE': 'see', 'LOOK': 'look', 'WATCH': 'watch', 'HEAR': 'hear',
            'EAT': 'eat', 'DRINK': 'drink', 'SLEEP': 'sleep',
            'WORK': 'work', 'PLAY': 'play', 'STUDY': 'study',
            'BUY': 'buy', 'SELL': 'sell', 'PAY': 'pay', 'COST': 'cost',
            'GIVE': 'give', 'TAKE': 'take', 'GET': 'get', 'HAVE': 'have',
            'MAKE': 'make', 'DO': 'do', 'HELP': 'help',
            'KNOW': 'know', 'UNDERSTAND': 'understand', 'THINK': 'think',
            'FEEL': 'feel', 'HOPE': 'hope', 'BELIEVE': 'believe',
            'SAY': 'say', 'TELL': 'tell', 'TALK': 'talk', 'SPEAK': 'speak',
            'READ': 'read', 'WRITE': 'write', 'LEARN': 'learn', 'TEACH': 'teach',
            'DRIVE': 'drive', 'WALK': 'walk', 'RUN': 'run', 'SIT': 'sit',
            'STAND': 'stand', 'OPEN': 'open', 'CLOSE': 'close',
            'START': 'start', 'STOP': 'stop', 'FINISH': 'finish',
            'WIN': 'win', 'LOSE': 'lose', 'TRY': 'try',
            
            # Modal and auxiliary verbs
            'CAN': 'can', 'CANNOT': 'cannot', 'MUST': 'must',
            'SHOULD': 'should', 'WILL': 'will', 'WOULD': 'would',
            'COULD': 'could', 'MAY': 'may', 'MIGHT': 'might',
            
            # Common nouns
            'STORE': 'store', 'SHOP': 'shop', 'MARKET': 'market',
            'HOUSE': 'house', 'HOME': 'home', 'BUILDING': 'building',
            'SCHOOL': 'school', 'UNIVERSITY': 'university', 'COLLEGE': 'college',
            'HOSPITAL': 'hospital', 'DOCTOR': 'doctor', 'NURSE': 'nurse',
            'RESTAURANT': 'restaurant', 'HOTEL': 'hotel', 'BANK': 'bank',
            'LIBRARY': 'library', 'MUSEUM': 'museum', 'PARK': 'park',
            'CAR': 'car', 'BUS': 'bus', 'TRAIN': 'train', 'PLANE': 'plane',
            'BOOK': 'book', 'COMPUTER': 'computer', 'PHONE': 'phone',
            'MONEY': 'money', 'DOLLAR': 'dollar', 'CREDIT-CARD': 'credit card',
            'FOOD': 'food', 'WATER': 'water', 'COFFEE': 'coffee', 'TEA': 'tea',
            'APPLE': 'apple', 'BANANA': 'banana', 'BREAD': 'bread', 'MILK': 'milk',
            'FRIEND': 'friend', 'FAMILY': 'family', 'MOTHER': 'mother',
            'FATHER': 'father', 'SISTER': 'sister', 'BROTHER': 'brother',
            'WIFE': 'wife', 'HUSBAND': 'husband', 'CHILD': 'child',
            'BOY': 'boy', 'GIRL': 'girl', 'MAN': 'man', 'WOMAN': 'woman',
            'PERSON': 'person', 'PEOPLE': 'people',
            'DOG': 'dog', 'CAT': 'cat', 'ANIMAL': 'animal',
            'TIME': 'time', 'DAY': 'day', 'WEEK': 'week', 'MONTH': 'month',
            'YEAR': 'year', 'HOUR': 'hour', 'MINUTE': 'minute',
            'MORNING': 'morning', 'AFTERNOON': 'afternoon', 'EVENING': 'evening',
            'NIGHT': 'night', 'TODAY': 'today', 'TOMORROW': 'tomorrow',
            'YESTERDAY': 'yesterday', 'NOW': 'now', 'LATER': 'later',
            
            # Adjectives
            'GOOD': 'good', 'BAD': 'bad', 'GREAT': 'great', 'TERRIBLE': 'terrible',
            'BIG': 'big', 'SMALL': 'small', 'LARGE': 'large', 'TINY': 'tiny',
            'TALL': 'tall', 'SHORT': 'short', 'LONG': 'long',
            'HOT': 'hot', 'COLD': 'cold', 'WARM': 'warm', 'COOL': 'cool',
            'NEW': 'new', 'OLD': 'old', 'YOUNG': 'young',
            'FAST': 'fast', 'SLOW': 'slow', 'QUICK': 'quick',
            'EASY': 'easy', 'HARD': 'hard', 'DIFFICULT': 'difficult',
            'CHEAP': 'cheap', 'EXPENSIVE': 'expensive',
            'HAPPY': 'happy', 'SAD': 'sad', 'ANGRY': 'angry',
            'TIRED': 'tired', 'HUNGRY': 'hungry', 'THIRSTY': 'thirsty',
            'SICK': 'sick', 'HEALTHY': 'healthy',
            'BEAUTIFUL': 'beautiful', 'UGLY': 'ugly', 'PRETTY': 'pretty',
            'CLEAN': 'clean', 'DIRTY': 'dirty',
            'FULL': 'full', 'EMPTY': 'empty',
            'BUSY': 'busy', 'FREE': 'free',
            'IMPORTANT': 'important', 'INTERESTING': 'interesting',
            
            # Colors
            'RED': 'red', 'BLUE': 'blue', 'GREEN': 'green', 'YELLOW': 'yellow',
            'BLACK': 'black', 'WHITE': 'white', 'BROWN': 'brown',
            'ORANGE': 'orange', 'PURPLE': 'purple', 'PINK': 'pink',
            
            # Numbers
            'ONE': 'one', 'TWO': 'two', 'THREE': 'three', 'FOUR': 'four',
            'FIVE': 'five', 'SIX': 'six', 'SEVEN': 'seven', 'EIGHT': 'eight',
            'NINE': 'nine', 'TEN': 'ten', 'MANY': 'many', 'FEW': 'few',
            'SOME': 'some', 'ALL': 'all', 'NONE': 'none',
            
            # Prepositions and conjunctions
            'WITH': 'with', 'WITHOUT': 'without', 'FOR': 'for', 'FROM': 'from',
            'TO': 'to', 'AT': 'at', 'IN': 'in', 'ON': 'on', 'UNDER': 'under',
            'OVER': 'over', 'ABOVE': 'above', 'BELOW': 'below',
            'NEAR': 'near', 'FAR': 'far', 'NEXT-TO': 'next to',
            'BEFORE': 'before', 'AFTER': 'after', 'DURING': 'during',
            'AND': 'and', 'OR': 'or', 'BUT': 'but', 'BECAUSE': 'because',
            'IF': 'if', 'THEN': 'then', 'SO': 'so',
            
            # Negation and affirmation
            'NOT': 'not', 'NO': 'no', 'YES': 'yes', 'MAYBE': 'maybe',
            'SURE': 'sure', 'POSSIBLE': 'possible', 'IMPOSSIBLE': 'impossible',
            
            # Common phrases/expressions
            'THANK-YOU': 'thank you', 'PLEASE': 'please', 'SORRY': 'sorry',
            'EXCUSE-ME': 'excuse me', 'WELCOME': 'welcome',
            'HELLO': 'hello', 'GOODBYE': 'goodbye', 'SEE-YOU-LATER': 'see you later',
            'NICE-MEET-YOU': 'nice to meet you', 'HOW-ARE-YOU': 'how are you',
            'FINE': 'fine', 'OK': 'okay', 'ALRIGHT': 'alright',
            
            # Quantities and measures
            'MUCH': 'much', 'LITTLE': 'little', 'MORE': 'more', 'LESS': 'less',
            'MOST': 'most', 'LEAST': 'least', 'ENOUGH': 'enough',
            'TOO-MUCH': 'too much', 'TOO-LITTLE': 'too little',
        }
        
        # Temporal markers that indicate tense
        self.temporal_markers = {
            'YESTERDAY': ('past', 'yesterday'),
            'TOMORROW': ('future', 'tomorrow'),
            'PAST': ('past', ''),
            'FUTURE': ('future', ''),
            'WILL': ('future', 'will'),
            'FINISH': ('past', ''),  # ASL perfective marker
            'DONE': ('past', 'done'),
        }
    
    def clean_gloss(self, gloss: str) -> str:
        """
        Clean and normalize ASL gloss input.
        
        Args:
            gloss: Raw ASL gloss string
            
        Returns:
            Cleaned gloss string
        """
        # Convert to uppercase and normalize whitespace
        gloss = re.sub(r'\s+', ' ', gloss.strip().upper())
        
        # Remove common ASL notation that we don't need
        # Remove indexing (IX-1, IX-2, etc.)
        gloss = re.sub(r'IX-\d+', '', gloss)
        
        # Remove classifier information in parentheses
        gloss = re.sub(r'\([^)]*\)', '', gloss)
        
        # Remove non-manual markers (we'll handle these as context later)
        gloss = re.sub(r'[a-z]+:', '', gloss)  # removes things like "t:", "q:", "hs:"
        
        # Clean up extra spaces
        gloss = re.sub(r'\s+', ' ', gloss.strip())
        
        return gloss
    
    def tokenize_gloss(self, gloss: str) -> List[str]:
        """
        Split gloss into individual signs.
        
        Args:
            gloss: Cleaned ASL gloss
            
        Returns:
            List of individual signs
        """
        # Handle compound signs with hyphens
        signs = []
        for token in gloss.split():
            if token:  # Skip empty tokens
                signs.append(token)
        
        return signs
    
    def detect_temporal_context(self, signs: List[str]) -> Tuple[str, List[str]]:
        """
        Detect tense from temporal markers and remove them from the sign list.
        
        Args:
            signs: List of ASL signs
            
        Returns:
            Tuple of (tense, cleaned_signs)
            tense: 'past', 'present', or 'future'
            cleaned_signs: signs with temporal markers removed
        """
        tense = 'present'  # default
        cleaned_signs = []
        
        for sign in signs:
            if sign in self.temporal_markers:
                marker_tense, word = self.temporal_markers[sign]
                tense = marker_tense
                # Only add the word if it's not empty (some markers are just grammatical)
                if word:
                    cleaned_signs.append(word.upper())
            else:
                cleaned_signs.append(sign)
        
        return tense, cleaned_signs
    
    def translate_sign(self, sign: str) -> str:
        """
        Translate a single ASL sign to English word(s).
        
        Args:
            sign: Individual ASL sign
            
        Returns:
            English translation
        """
        # Handle compound signs (connected with hyphens)
        if '-' in sign:
            parts = sign.split('-')
            translated_parts = []
            for part in parts:
                if part in self.sign_dictionary:
                    translated_parts.append(self.sign_dictionary[part])
                else:
                    translated_parts.append(part.lower())
            return ' '.join(translated_parts)
        
        # Direct dictionary lookup
        if sign in self.sign_dictionary:
            return self.sign_dictionary[sign]
        
        # Handle numbers (1, 2, 3, etc.) - convert digits to words
        if sign.isdigit():
            digit_words = {
                '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
                '10': 'ten', '11': 'eleven', '12': 'twelve'
            }
            return digit_words.get(sign, sign)
        
        # If not found, return lowercase version (fallback)
        return sign.lower()
    
    def basic_word_order_fix(self, words: List[str], tense: str) -> List[str]:
        """
        Apply basic English word order rules.
        
        ASL often uses different word orders than English. This function
        applies some basic fixes, but the grammar corrector will handle
        more complex restructuring.
        
        Args:
            words: List of English words from sign translation
            tense: Detected tense ('past', 'present', 'future')
            
        Returns:
            Reordered words
        """
        if len(words) < 2:
            return words
        
        # Simple pronoun case fixing (I vs me in subject position)
        if words and words[0] == 'me':
            words[0] = 'I'
        
        # Handle negation - move 'not' closer to verbs
        if 'not' in words:
            # This is a simple approach - the grammar model will refine it
            not_index = words.index('not')
            # Look for a verb after 'not'
            for i in range(not_index + 1, len(words)):
                if words[i] in ['want', 'need', 'like', 'go', 'come', 'have', 'can', 'will']:
                    # Move 'not' to just before the verb
                    words.pop(not_index)
                    words.insert(i - 1, 'not')
                    break
        
        return words
    
    def translate(self, gloss: str) -> Dict[str, any]:
        """
        Main translation function that converts ASL gloss to rough English.
        
        Args:
            gloss: Raw ASL gloss input
            
        Returns:
            Dictionary containing translation results and metadata
        """
        # Step 1: Clean the input
        cleaned_gloss = self.clean_gloss(gloss)
        
        # Step 2: Tokenize into signs
        signs = self.tokenize_gloss(cleaned_gloss)
        
        # Step 3: Detect temporal context
        tense, signs_without_temporal = self.detect_temporal_context(signs)
        
        # Step 4: Translate each sign
        english_words = []
        for sign in signs_without_temporal:
            translated = self.translate_sign(sign)
            english_words.append(translated)
        
        # Step 5: Apply basic word order fixes
        ordered_words = self.basic_word_order_fix(english_words, tense)
        
        # Step 6: Join into rough English sentence
        rough_english = ' '.join(ordered_words)
        
        # Add basic capitalization and punctuation
        if rough_english:
            rough_english = rough_english[0].upper() + rough_english[1:] + '.'
        
        return {
            'original_gloss': gloss,
            'cleaned_gloss': cleaned_gloss,
            'signs': signs,
            'detected_tense': tense,
            'rough_english': rough_english,
            'word_count': len(ordered_words)
        }


# Example usage and testing
if __name__ == "__main__":
    translator = RuleBasedTranslator()
    
    # Test cases
    test_glosses = [
        "ME WANT GO STORE BUY APPLE",
        "YESTERDAY YOU EAT PIZZA",
        "WHERE YOU LIVE",
        "ME NOT LIKE COLD WEATHER",
        "TOMORROW WE MEET FRIEND RESTAURANT",
        "BOOK RED VERY INTERESTING",
        "HOW MUCH COST CAR",
        "ME FINISH HOMEWORK ALREADY",
        "CAN YOU HELP ME PLEASE",
        "COFFEE HOT BUT GOOD"
    ]
    
    print("Rule-Based ASL Translator Test")
    print("=" * 50)
    
    for gloss in test_glosses:
        result = translator.translate(gloss)
        print(f"Original: {result['original_gloss']}")
        print(f"Rough English: {result['rough_english']}")
        print(f"Detected Tense: {result['detected_tense']}")
        print("-" * 30)