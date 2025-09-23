import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Word:
    text: str
    pos: str  # part of speech
    role: str = ""  # grammatical role (subject, object, etc.)

class ASLTranslator:
    def __init__(self):
        # Enhanced dictionary with part-of-speech tagging
        self.sign_dictionary = {
            # Pronouns (subject/object forms)
            'ME': {'word': 'I', 'pos': 'pronoun', 'obj_form': 'me'},
            'YOU': {'word': 'you', 'pos': 'pronoun', 'obj_form': 'you'},
            'HE': {'word': 'he', 'pos': 'pronoun', 'obj_form': 'him'},
            'SHE': {'word': 'she', 'pos': 'pronoun', 'obj_form': 'her'},
            'IT': {'word': 'it', 'pos': 'pronoun', 'obj_form': 'it'},
            'WE': {'word': 'we', 'pos': 'pronoun', 'obj_form': 'us'},
            'THEY': {'word': 'they', 'pos': 'pronoun', 'obj_form': 'them'},
            
            # Possessives
            'MY': {'word': 'my', 'pos': 'determiner'},
            'YOUR': {'word': 'your', 'pos': 'determiner'},
            'HIS': {'word': 'his', 'pos': 'determiner'},
            'HER': {'word': 'her', 'pos': 'determiner'},
            'THEIR': {'word': 'their', 'pos': 'determiner'},
            
            # Question words
            'WHO': {'word': 'who', 'pos': 'wh-pronoun'},
            'WHAT': {'word': 'what', 'pos': 'wh-pronoun'},
            'WHERE': {'word': 'where', 'pos': 'wh-adverb'},
            'WHEN': {'word': 'when', 'pos': 'wh-adverb'},
            'WHY': {'word': 'why', 'pos': 'wh-adverb'},
            'HOW': {'word': 'how', 'pos': 'wh-adverb'},
            'WHICH': {'word': 'which', 'pos': 'wh-determiner'},
            
            # Modal verbs
            'CAN': {'word': 'can', 'pos': 'modal'},
            'CANNOT': {'word': 'cannot', 'pos': 'modal'},
            'MUST': {'word': 'must', 'pos': 'modal'},
            'SHOULD': {'word': 'should', 'pos': 'modal'},
            'WILL': {'word': 'will', 'pos': 'modal'},
            'WOULD': {'word': 'would', 'pos': 'modal'},
            'COULD': {'word': 'could', 'pos': 'modal'},
            
            # Verbs
            'GO': {'word': 'go', 'pos': 'verb', 'past': 'went', '3rd': 'goes'},
            'COME': {'word': 'come', 'pos': 'verb', 'past': 'came', '3rd': 'comes'},
            'SEE': {'word': 'see', 'pos': 'verb', 'past': 'saw', '3rd': 'sees'},
            'WATCH': {'word': 'watch', 'pos': 'verb', 'past': 'watched', '3rd': 'watches'},
            'LOOK': {'word': 'look', 'pos': 'verb', 'past': 'looked', '3rd': 'looks'},
            'HEAR': {'word': 'hear', 'pos': 'verb', 'past': 'heard', '3rd': 'hears'},
            'EAT': {'word': 'eat', 'pos': 'verb', 'past': 'ate', '3rd': 'eats'},
            'DRINK': {'word': 'drink', 'pos': 'verb', 'past': 'drank', '3rd': 'drinks'},
            'BUY': {'word': 'buy', 'pos': 'verb', 'past': 'bought', '3rd': 'buys'},
            'GIVE': {'word': 'give', 'pos': 'verb', 'past': 'gave', '3rd': 'gives'},
            'TAKE': {'word': 'take', 'pos': 'verb', 'past': 'took', '3rd': 'takes'},
            'HELP': {'word': 'help', 'pos': 'verb', 'past': 'helped', '3rd': 'helps'},
            'WANT': {'word': 'want', 'pos': 'verb', 'past': 'wanted', '3rd': 'wants'},
            'NEED': {'word': 'need', 'pos': 'verb', 'past': 'needed', '3rd': 'needs'},
            'LIKE': {'word': 'like', 'pos': 'verb', 'past': 'liked', '3rd': 'likes'},
            'LOVE': {'word': 'love', 'pos': 'verb', 'past': 'loved', '3rd': 'loves'},
            'KNOW': {'word': 'know', 'pos': 'verb', 'past': 'knew', '3rd': 'knows'},
            'THINK': {'word': 'think', 'pos': 'verb', 'past': 'thought', '3rd': 'thinks'},
            'FEEL': {'word': 'feel', 'pos': 'verb', 'past': 'felt', '3rd': 'feels'},
            'WORK': {'word': 'work', 'pos': 'verb', 'past': 'worked', '3rd': 'works'},
            'STUDY': {'word': 'study', 'pos': 'verb', 'past': 'studied', '3rd': 'studies'},
            'TEACH': {'word': 'teach', 'pos': 'verb', 'past': 'taught', '3rd': 'teaches'},
            'LEARN': {'word': 'learn', 'pos': 'verb', 'past': 'learned', '3rd': 'learns'},
            'READ': {'word': 'read', 'pos': 'verb', 'past': 'read', '3rd': 'reads'},
            'WRITE': {'word': 'write', 'pos': 'verb', 'past': 'wrote', '3rd': 'writes'},
            'TALK': {'word': 'talk', 'pos': 'verb', 'past': 'talked', '3rd': 'talks'},
            'SAY': {'word': 'say', 'pos': 'verb', 'past': 'said', '3rd': 'says'},
            'TELL': {'word': 'tell', 'pos': 'verb', 'past': 'told', '3rd': 'tells'},
            'WALK': {'word': 'walk', 'pos': 'verb', 'past': 'walked', '3rd': 'walks'},
            'SIT': {'word': 'sit', 'pos': 'verb', 'past': 'sat', '3rd': 'sits'},
            'LAUGH': {'word': 'laugh', 'pos': 'verb', 'past': 'laughed', '3rd': 'laughs'},
            'DISCUSS': {'word': 'discuss', 'pos': 'verb', 'past': 'discussed', '3rd': 'discusses'},
            
            # Nouns
            'MOVIE': {'word': 'movie', 'pos': 'noun', 'countable': True},
            'FILM': {'word': 'film', 'pos': 'noun', 'countable': True},
            'FILMS': {'word': 'films', 'pos': 'noun', 'plural': True},
            'FESTIVAL': {'word': 'festival', 'pos': 'noun', 'countable': True},
            'CINEMA': {'word': 'cinema', 'pos': 'noun', 'countable': True},
            'LINE': {'word': 'line', 'pos': 'noun', 'countable': True},
            'POPCORN': {'word': 'popcorn', 'pos': 'noun', 'countable': False},
            'SODA': {'word': 'soda', 'pos': 'noun', 'countable': True},
            'FRIEND': {'word': 'friend', 'pos': 'noun', 'countable': True},
            'FRIENDS': {'word': 'friends', 'pos': 'noun', 'plural': True},
            'STORE': {'word': 'store', 'pos': 'noun', 'countable': True},
            'PARK': {'word': 'park', 'pos': 'noun', 'countable': True},
            'APPLE': {'word': 'apple', 'pos': 'noun', 'countable': True},
            'BANANA': {'word': 'banana', 'pos': 'noun', 'countable': True},
            'MILK': {'word': 'milk', 'pos': 'noun', 'countable': False},
            'SUNSET': {'word': 'sunset', 'pos': 'noun', 'countable': True},
            'HOUSE': {'word': 'house', 'pos': 'noun', 'countable': True},
            'CAR': {'word': 'car', 'pos': 'noun', 'countable': True},
            'BOOK': {'word': 'book', 'pos': 'noun', 'countable': True},
            'SCHOOL': {'word': 'school', 'pos': 'noun', 'countable': True},
            'HOME': {'word': 'home', 'pos': 'noun', 'countable': True},
            'FAMILY': {'word': 'family', 'pos': 'noun', 'countable': True},
            'TIME': {'word': 'time', 'pos': 'noun', 'countable': False},
            'NIGHT': {'word': 'night', 'pos': 'noun', 'countable': True},
            
            # Adjectives
            'BEAUTIFUL': {'word': 'beautiful', 'pos': 'adjective'},
            'GOOD': {'word': 'good', 'pos': 'adjective'},
            'BAD': {'word': 'bad', 'pos': 'adjective'},
            'BIG': {'word': 'big', 'pos': 'adjective'},
            'SMALL': {'word': 'small', 'pos': 'adjective'},
            'HOT': {'word': 'hot', 'pos': 'adjective'},
            'COLD': {'word': 'cold', 'pos': 'adjective'},
            'OLD': {'word': 'old', 'pos': 'adjective'},
            'NEW': {'word': 'new', 'pos': 'adjective'},
            'HAPPY': {'word': 'happy', 'pos': 'adjective'},
            'SAD': {'word': 'sad', 'pos': 'adjective'},
            'HUNGRY': {'word': 'hungry', 'pos': 'adjective'},
            'TIRED': {'word': 'tired', 'pos': 'adjective'},
            
            # Temporal markers
            'YESTERDAY': {'word': 'yesterday', 'pos': 'adverb', 'tense': 'past'},
            'TOMORROW': {'word': 'tomorrow', 'pos': 'adverb', 'tense': 'future'},
            'TODAY': {'word': 'today', 'pos': 'adverb', 'tense': 'present'},
            'NOW': {'word': 'now', 'pos': 'adverb', 'tense': 'present'},
            'BEFORE': {'word': 'before', 'pos': 'adverb', 'tense': 'past'},
            'AFTER': {'word': 'after', 'pos': 'adverb'},
            'FINISH': {'word': 'finished', 'pos': 'verb', 'tense': 'past'},
            'DONE': {'word': 'done', 'pos': 'adjective', 'tense': 'past'},
            'PAST': {'word': 'in the past', 'pos': 'adverb', 'tense': 'past'},
            'FUTURE': {'word': 'in the future', 'pos': 'adverb', 'tense': 'future'},
            
            # Prepositions and conjunctions
            'WITH': {'word': 'with', 'pos': 'preposition'},
            'AND': {'word': 'and', 'pos': 'conjunction'},
            'THAT': {'word': 'that', 'pos': 'conjunction'},
            'BUT': {'word': 'but', 'pos': 'conjunction'},
            'OR': {'word': 'or', 'pos': 'conjunction'},
            'TO': {'word': 'to', 'pos': 'preposition'},
            'AT': {'word': 'at', 'pos': 'preposition'},
            'IN': {'word': 'in', 'pos': 'preposition'},
            'ON': {'word': 'on', 'pos': 'preposition'},
            
            # Negation
            'NOT': {'word': 'not', 'pos': 'adverb'},
            'NO': {'word': 'no', 'pos': 'determiner'},
            'NEVER': {'word': 'never', 'pos': 'adverb'},
        }
        
        # Common ASL sentence patterns
        self.patterns = [
            # Topic-comment structure: MOVIE FESTIVAL, ME GO
            {
                'pattern': r'^([A-Z\s]+),\s*([A-Z\s]+)$',
                'handler': self.handle_topic_comment
            },
            # Temporal sequence: YESTERDAY ... AFTER THAT ...
            {
                'pattern': r'(.*)\bAFTER\s+THAT\b(.*)',
                'handler': self.handle_temporal_sequence
            },
            # List structure: A B C D (multiple nouns/actions)
            {
                'pattern': r'^([A-Z]+\s+){4,}',
                'handler': self.handle_descriptive_list
            }
        ]
    
    def parse_gloss(self, gloss: str) -> List[str]:
        """Enhanced parsing with better handling of ASL notation"""
        # Clean up the gloss
        gloss = re.sub(r'\s+', ' ', gloss.strip().upper())
        
        # Remove ASL notation markers but preserve important ones
        gloss = re.sub(r'IX-\d+', '', gloss)  # Remove indexing
        gloss = re.sub(r'\([^)]*\)', '', gloss)  # Remove classifiers
        gloss = re.sub(r'hs:', '', gloss)  # Remove head shake
        gloss = re.sub(r't:', 'TOPIC:', gloss)  # Mark topics
        gloss = re.sub(r'q:', 'QUESTION:', gloss)  # Mark questions
        
        signs = [sign.strip() for sign in gloss.split() if sign.strip()]
        return signs
    
    def identify_sentence_structure(self, signs: List[str]) -> str:
        """Identify the type of sentence structure"""
        # Check for temporal sequences
        if 'AFTER' in signs and 'THAT' in signs:
            return 'temporal_sequence'
        
        # Check for topic-comment (comma or pause indicators)
        if len(signs) > 6 and any(pos in ['noun', 'adjective'] 
                                 for pos in [self.get_pos(sign) for sign in signs[:3]]):
            return 'topic_comment'
        
        # Check for simple subject-verb-object
        if len(signs) <= 5:
            return 'simple_svo'
        
        # Check for list/description (many nouns/adjectives in sequence)
        noun_count = sum(1 for sign in signs if self.get_pos(sign) == 'noun')
        if noun_count > 3:
            return 'descriptive_list'
        
        return 'complex'
    
    def get_pos(self, sign: str) -> str:
        """Get part of speech for a sign"""
        if sign in self.sign_dictionary:
            return self.sign_dictionary[sign]['pos']
        return 'unknown'
    
    def translate_sign(self, sign: str, role: str = 'subject') -> Word:
        """Enhanced sign translation with grammatical role consideration"""
        if sign in self.sign_dictionary:
            entry = self.sign_dictionary[sign]
            word_text = entry['word']
            
            # Handle pronoun cases (subject vs object)
            if entry['pos'] == 'pronoun' and role == 'object' and 'obj_form' in entry:
                word_text = entry['obj_form']
            
            return Word(text=word_text, pos=entry['pos'], role=role)
        
        # Handle unknown signs
        return Word(text=sign.lower(), pos='unknown', role=role)
    
    def detect_tense_and_aspect(self, signs: List[str]) -> Tuple[str, str, List[str]]:
        """Enhanced tense and aspect detection"""
        tense = 'present'
        aspect = 'simple'  # simple, progressive, perfect
        cleaned_signs = []
        
        temporal_markers = {
            'past': ['YESTERDAY', 'BEFORE', 'PAST', 'FINISH', 'DONE'],
            'future': ['TOMORROW', 'WILL', 'FUTURE', 'LATER'],
            'present': ['NOW', 'TODAY']
        }
        
        for sign in signs:
            is_temporal = False
            for tense_type, markers in temporal_markers.items():
                if sign in markers:
                    if sign == 'FINISH':
                        aspect = 'perfect'
                    tense = tense_type
                    is_temporal = True
                    if sign not in ['NOW', 'TODAY']:  # Keep present markers
                        break
            
            if not is_temporal or sign in ['NOW', 'TODAY']:
                cleaned_signs.append(sign)
        
        return tense, aspect, cleaned_signs
    
    def handle_temporal_sequence(self, signs: List[str]) -> List[Word]:
        """Handle sentences with temporal sequences like 'AFTER THAT'"""
        # Split at AFTER THAT
        after_idx = None
        for i, sign in enumerate(signs):
            if i < len(signs) - 1 and sign == 'AFTER' and signs[i + 1] == 'THAT':
                after_idx = i
                break
        
        if after_idx is None:
            return self.basic_translation(signs)
        
        # Process first part
        first_part = signs[:after_idx]
        second_part = signs[after_idx + 2:]  # Skip AFTER THAT
        
        # Translate both parts
        first_words = self.basic_translation(first_part)
        second_words = self.basic_translation(second_part)
        
        # Connect with appropriate conjunction
        connector = Word(text='and then', pos='conjunction')
        
        return first_words + [connector] + second_words
    
    def handle_topic_comment(self, signs: List[str]) -> List[Word]:
        """Handle topic-comment structure"""
        # Find natural break point (usually after 2-3 nouns/adjectives)
        break_point = 0
        noun_adj_count = 0
        
        for i, sign in enumerate(signs):
            pos = self.get_pos(sign)
            if pos in ['noun', 'adjective']:
                noun_adj_count += 1
                if noun_adj_count >= 2:
                    break_point = i + 1
                    break
            elif pos in ['pronoun', 'verb']:
                break_point = i
                break
        
        if break_point == 0:
            return self.basic_translation(signs)
        
        topic = signs[:break_point]
        comment = signs[break_point:]
        
        # Translate parts
        topic_words = self.basic_translation(topic)
        comment_words = self.basic_translation(comment)
        
        # Restructure: "At the movie festival, I went..." or "The movie festival was..."
        if comment_words and comment_words[0].pos == 'pronoun':
            # Add "At the" or "During the" to topic
            prep = Word(text='At the', pos='preposition')
            return [prep] + topic_words + [Word(text=',', pos='punctuation')] + comment_words
        else:
            # Make topic the subject
            return topic_words + comment_words
    
    def handle_descriptive_list(self, signs: List[str]) -> List[Word]:
        """Handle lists of items/actions"""
        words = []
        current_group = []
        
        for i, sign in enumerate(signs):
            word = self.translate_sign(sign)
            pos = word.pos
            
            if pos in ['noun', 'adjective']:
                current_group.append(word)
            elif pos == 'verb':
                # End current group, add conjunctions
                if current_group:
                    words.extend(self.add_conjunctions(current_group))
                    current_group = []
                words.append(word)
            else:
                if current_group:
                    words.extend(self.add_conjunctions(current_group))
                    current_group = []
                words.append(word)
        
        # Handle remaining group
        if current_group:
            words.extend(self.add_conjunctions(current_group))
        
        return words
    
    def add_conjunctions(self, words: List[Word]) -> List[Word]:
        """Add appropriate conjunctions between words"""
        if len(words) <= 1:
            return words
        
        result = []
        for i, word in enumerate(words):
            if i == len(words) - 1 and i > 0:
                # Last item: "and X"
                result.append(Word(text='and', pos='conjunction'))
            elif i > 0:
                # Middle items: ", X"
                result.append(Word(text=',', pos='punctuation'))
            
            result.append(word)
        
        return result
    
    def basic_translation(self, signs: List[str]) -> List[Word]:
        """Basic word-by-word translation with role assignment"""
        words = []
        
        # Simple role assignment based on position
        for i, sign in enumerate(signs):
            if i == 0 and self.get_pos(sign) == 'pronoun':
                role = 'subject'
            elif self.get_pos(sign) == 'pronoun' and i > 0:
                # Look for verb before this pronoun
                has_verb_before = any(self.get_pos(signs[j]) in ['verb', 'modal'] 
                                    for j in range(i))
                role = 'object' if has_verb_before else 'subject'
            else:
                role = 'neutral'
            
            words.append(self.translate_sign(sign, role))
        
        return words
    
    def apply_grammar_rules(self, words: List[Word], tense: str, is_question: bool) -> List[Word]:
        """Apply comprehensive grammar rules"""
        if not words:
            return words
        
        # 1. Handle subject-verb agreement and conjugation
        words = self.handle_verb_conjugation(words, tense)
        
        # 2. Fix pronoun cases
        words = self.fix_pronoun_cases(words)
        
        # 3. Add articles
        words = self.add_articles_enhanced(words)
        
        # 4. Handle negation properly
        words = self.handle_negation_enhanced(words)
        
        # 5. Reorder for questions
        if is_question:
            words = self.reorder_for_questions(words)
        
        # 6. Add missing copula (be verbs)
        words = self.add_copula(words)
        
        return words
    
    def handle_verb_conjugation(self, words: List[Word], tense: str) -> List[Word]:
        """Enhanced verb conjugation"""
        result = []
        subject = None
        
        for i, word in enumerate(words):
            if word.pos == 'pronoun' and word.role == 'subject':
                subject = word.text.lower()
            elif word.pos in ['verb', 'modal']:
                if word.pos == 'verb' and subject:
                    # Conjugate the verb
                    sign_key = None
                    for key, entry in self.sign_dictionary.items():
                        if entry.get('word') == word.text:
                            sign_key = key
                            break
                    
                    if sign_key and sign_key in self.sign_dictionary:
                        entry = self.sign_dictionary[sign_key]
                        if tense == 'past' and 'past' in entry:
                            word.text = entry['past']
                        elif tense == 'present' and subject in ['he', 'she', 'it'] and '3rd' in entry:
                            word.text = entry['3rd']
                        elif tense == 'future':
                            # Add 'will' before verb if not already there
                            if i == 0 or words[i-1].text != 'will':
                                result.append(Word(text='will', pos='modal'))
            
            result.append(word)
        
        return result
    
    def fix_pronoun_cases(self, words: List[Word]) -> List[Word]:
        """Fix pronoun cases (I/me, he/him, etc.)"""
        for i, word in enumerate(words):
            if word.pos == 'pronoun':
                # Look for context clues
                is_object = False
                
                # Check if after verb or preposition
                if i > 0:
                    prev_pos = words[i-1].pos
                    if prev_pos in ['verb', 'preposition', 'modal']:
                        is_object = True
                
                # Check if after helping verb
                if i > 1 and words[i-1].pos == 'verb' and words[i-2].pos == 'modal':
                    is_object = True
                
                # Apply object form if needed
                if is_object:
                    for key, entry in self.sign_dictionary.items():
                        if (entry.get('word', '').lower() == word.text.lower() and 
                            entry.get('pos') == 'pronoun' and 'obj_form' in entry):
                            word.text = entry['obj_form']
                            break
        
        return words
    
    def add_articles_enhanced(self, words: List[Word]) -> List[Word]:
        """Enhanced article addition"""
        result = []
        i = 0
        
        while i < len(words):
            word = words[i]
            
            # Check if we need an article before a noun
            if (word.pos == 'noun' and 
                not word.text.endswith('s') and  # Not plural
                word.text not in ['milk', 'water', 'coffee', 'popcorn'] and  # Not uncountable
                (i == 0 or words[i-1].pos not in ['determiner', 'adjective'])):
                
                # Determine article
                if word.text[0].lower() in 'aeiou':
                    result.append(Word(text='an', pos='determiner'))
                else:
                    result.append(Word(text='a', pos='determiner'))
            
            result.append(word)
            i += 1
        
        return result
    
    def handle_negation_enhanced(self, words: List[Word]) -> List[Word]:
        """Enhanced negation handling"""
        if not any(word.text == 'not' for word in words):
            return words
        
        result = []
        not_word = None
        
        for word in words:
            if word.text == 'not':
                not_word = word
                continue
            elif word.pos in ['modal', 'verb'] and not_word:
                result.append(word)
                result.append(not_word)
                not_word = None
            else:
                result.append(word)
        
        # If 'not' wasn't placed, add it at the end
        if not_word:
            result.append(not_word)
        
        return result
    
    def add_copula(self, words: List[Word]) -> List[Word]:
        """Add missing 'be' verbs"""
        if len(words) < 2:
            return words
        
        result = []
        i = 0
        
        while i < len(words):
            word = words[i]
            result.append(word)
            
            # Check if we need a copula between pronoun and adjective/noun
            if (i < len(words) - 1 and 
                word.pos == 'pronoun' and 
                words[i + 1].pos in ['adjective', 'noun'] and
                word.role == 'subject'):
                
                # Add appropriate form of 'be'
                subject = word.text.lower()
                if subject == 'i':
                    result.append(Word(text='am', pos='verb'))
                elif subject in ['he', 'she', 'it']:
                    result.append(Word(text='is', pos='verb'))
                else:
                    result.append(Word(text='are', pos='verb'))
            
            i += 1
        
        return result
    
    def reorder_for_questions(self, words: List[Word]) -> List[Word]:
        """Reorder words for question formation"""
        wh_words = ['who', 'what', 'where', 'when', 'why', 'how', 'which']
        
        # Find WH-word
        wh_idx = None
        for i, word in enumerate(words):
            if word.text.lower() in wh_words:
                wh_idx = i
                break
        
        if wh_idx is not None and wh_idx != 0:
            # Move WH-word to front
            wh_word = words.pop(wh_idx)
            words.insert(0, wh_word)
        
        return words
    
    def detect_question(self, signs: List[str]) -> bool:
        """Enhanced question detection"""
        wh_words = ['WHO', 'WHAT', 'WHERE', 'WHEN', 'WHY', 'HOW', 'WHICH']
        return any(sign in wh_words for sign in signs)
    
    def translate(self, gloss: str) -> str:
        """Enhanced main translation function"""
        # Parse the gloss
        signs = self.parse_gloss(gloss)
        
        if not signs:
            return ""
        
        # Detect sentence structure
        structure_type = self.identify_sentence_structure(signs)
        
        # Detect tense and aspect
        tense, aspect, signs = self.detect_tense_and_aspect(signs)
        
        # Detect questions
        is_question = self.detect_question(signs)
        
        # Apply structure-specific handling
        if structure_type == 'temporal_sequence':
            words = self.handle_temporal_sequence(signs)
        elif structure_type == 'topic_comment':
            words = self.handle_topic_comment(signs)
        elif structure_type == 'descriptive_list':
            words = self.handle_descriptive_list(signs)
        else:
            words = self.basic_translation(signs)
        
        # Apply grammar rules
        words = self.apply_grammar_rules(words, tense, is_question)
        
        # Convert to text and add punctuation
        text_words = [word.text for word in words if word.text not in [',']]
        
        if not text_words:
            return ""
        
        # Capitalize first word
        text_words[0] = text_words[0].capitalize()
        
        # Join and punctuate
        result = ' '.join(text_words)
        
        # Clean up extra spaces
        result = re.sub(r'\s+', ' ', result).strip()
        
        # Add final punctuation
        if is_question and not result.endswith('?'):
            result += '?'
        elif not is_question and not result.endswith('.'):
            result += '.'
        
        return result

# Example usage and testing
if __name__ == "__main__":
    translator = ASLTranslator()
    
    # Enhanced test cases
    test_glosses = [
        # Basic sentences
        "ME GO SCHOOL",
        "YOU LIKE COFFEE", 
        "YESTERDAY ME SEE FRIEND",
        "YOU CAN HELP ME",  # Fixed pronoun case
        
        # Questions
        "WHAT YOU WANT",
        "WHERE YOU GO",
        "WHO COME YESTERDAY",
        
        # Negation
        "ME NOT HUNGRY",
        "YOU CANNOT GO",
        
        # Complex temporal sequences
        "YESTERDAY STORE I GO WITH FRIEND BUY APPLE BANANA MILK AFTER THAT PARK WE WALK SIT TALK LAUGH SUNSET BEAUTIFUL",
        
        # Topic-comment structures  
        "MOVIE FESTIVAL NIGHT CINEMA LINE POPCORN SODA WATCH FILMS DISCUSS FRIENDS",
        
        # Present and future
        "TOMORROW WE MEET FRIEND",
        "NOW ME STUDY BOOK",
        
        # Adjective predicates
        "SUNSET BEAUTIFUL",
        "HOUSE BIG",
        "FOOD GOOD",
        
        # Modal verbs
        "ME SHOULD GO HOME",
        "YOU MUST FINISH WORK",
        
        # More complex examples
        "YESTERDAY ME HAPPY BECAUSE FRIEND GIVE ME BOOK",
        "MOVIE GOOD BUT POPCORN BAD",
        "ME WANT GO STORE BUY APPLE AND BANANA",
    ]
    
    print("Enhanced ASL Gloss to English Translation System")
    print("=" * 60)
    
    for gloss in test_glosses:
        translation = translator.translate(gloss)
        print(f"ASL: {gloss}")
        print(f"English: {translation}")
        print("-" * 40)
    
    # Interactive mode
    print("\nInteractive Mode - Enter ASL glosses to translate:")
    print("(Type 'quit' to exit)")
    
    while True:
        user_input = input("\nASL Gloss: ").strip()
        if user_input.lower() == 'quit':
            break
        if user_input:
            translation = translator.translate(user_input)
            print(f"English: {translation}")
            
            # Show analysis for debugging
            print("Analysis:")
            signs = translator.parse_gloss(user_input)
            structure = translator.identify_sentence_structure(signs)
            tense, aspect, cleaned_signs = translator.detect_tense_and_aspect(signs)
            is_question = translator.detect_question(signs)
            
            print(f"  Structure: {structure}")
            print(f"  Tense: {tense}, Aspect: {aspect}")
            print(f"  Question: {is_question}")
            print(f"  Cleaned signs: {cleaned_signs}")