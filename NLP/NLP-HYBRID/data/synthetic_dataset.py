"""
Synthetic ASL Gloss to English Dataset Generator

Creates a comprehensive training dataset for ASL gloss translation,
including single sentences and paragraph-level examples.
"""

import json
import random
from typing import List, Tuple, Dict

class ASLGlossDataGenerator:
    """Generate synthetic ASL gloss to English translation pairs."""
    
    def __init__(self):
        # ASL gloss vocabulary and patterns
        self.pronouns = {
            "I": "I", "ME": "me", "YOU": "you", "HE": "he", "SHE": "she", 
            "WE": "we", "THEY": "they", "MY": "my", "YOUR": "your", 
            "HIS": "his", "HER": "her", "OUR": "our", "THEIR": "their"
        }
        
        self.time_markers = {
            "YESTERDAY": "yesterday", "TODAY": "today", "TOMORROW": "tomorrow",
            "MORNING": "morning", "AFTERNOON": "afternoon", "EVENING": "evening",
            "NIGHT": "night", "WEEK": "week", "MONTH": "month", "YEAR": "year",
            "PAST": "ago", "FUTURE": "will", "NOW": "now", "BEFORE": "before",
            "AFTER": "after", "LAST": "last", "NEXT": "next"
        }
        
        self.verbs = {
            "GO": ["go", "goes", "went", "going"],
            "COME": ["come", "comes", "came", "coming"],
            "EAT": ["eat", "eats", "ate", "eating"],
            "DRINK": ["drink", "drinks", "drank", "drinking"],
            "SLEEP": ["sleep", "sleeps", "slept", "sleeping"],
            "WORK": ["work", "works", "worked", "working"],
            "STUDY": ["study", "studies", "studied", "studying"],
            "READ": ["read", "reads", "read", "reading"],
            "WRITE": ["write", "writes", "wrote", "writing"],
            "PLAY": ["play", "plays", "played", "playing"],
            "WATCH": ["watch", "watches", "watched", "watching"],
            "LISTEN": ["listen", "listens", "listened", "listening"],
            "TALK": ["talk", "talks", "talked", "talking"],
            "WALK": ["walk", "walks", "walked", "walking"],
            "RUN": ["run", "runs", "ran", "running"],
            "DRIVE": ["drive", "drives", "drove", "driving"],
            "COOK": ["cook", "cooks", "cooked", "cooking"],
            "CLEAN": ["clean", "cleans", "cleaned", "cleaning"],
            "HELP": ["help", "helps", "helped", "helping"],
            "TEACH": ["teach", "teaches", "taught", "teaching"],
            "LEARN": ["learn", "learns", "learned", "learning"],
            "BUY": ["buy", "buys", "bought", "buying"],
            "SELL": ["sell", "sells", "sold", "selling"],
            "MEET": ["meet", "meets", "met", "meeting"],
            "CALL": ["call", "calls", "called", "calling"],
            "EMAIL": ["email", "emails", "emailed", "emailing"],
            "VISIT": ["visit", "visits", "visited", "visiting"],
            "TRAVEL": ["travel", "travels", "traveled", "traveling"],
            "SWIM": ["swim", "swims", "swam", "swimming"],
            "DANCE": ["dance", "dances", "danced", "dancing"],
            "SING": ["sing", "sings", "sang", "singing"]
        }
        
        self.nouns = {
            "STORE": "store", "SCHOOL": "school", "HOME": "home", "WORK": "work",
            "FRIEND": "friend", "FAMILY": "family", "MOTHER": "mother", "FATHER": "father",
            "BROTHER": "brother", "SISTER": "sister", "TEACHER": "teacher", "STUDENT": "student",
            "BOOK": "book", "COMPUTER": "computer", "PHONE": "phone", "CAR": "car",
            "HOUSE": "house", "APARTMENT": "apartment", "RESTAURANT": "restaurant",
            "HOSPITAL": "hospital", "LIBRARY": "library", "PARK": "park", "BEACH": "beach",
            "MOUNTAIN": "mountain", "CITY": "city", "COUNTRY": "country", "WORLD": "world",
            "COFFEE": "coffee", "TEA": "tea", "WATER": "water", "FOOD": "food",
            "BREAKFAST": "breakfast", "LUNCH": "lunch", "DINNER": "dinner",
            "MOVIE": "movie", "MUSIC": "music", "GAME": "game", "SPORT": "sport",
            "JOB": "job", "MONEY": "money", "TIME": "time", "DAY": "day",
            "WEEK": "week", "MONTH": "month", "YEAR": "year", "BIRTHDAY": "birthday",
            "PARTY": "party", "MEETING": "meeting", "CLASS": "class", "EXAM": "exam",
            "VACATION": "vacation", "TRIP": "trip", "WEATHER": "weather", "RAIN": "rain",
            "SUN": "sun", "SNOW": "snow", "WIND": "wind"
        }
        
        self.adjectives = {
            "GOOD": "good", "BAD": "bad", "HAPPY": "happy", "SAD": "sad",
            "BIG": "big", "SMALL": "small", "HOT": "hot", "COLD": "cold",
            "FAST": "fast", "SLOW": "slow", "EASY": "easy", "HARD": "hard",
            "BEAUTIFUL": "beautiful", "UGLY": "ugly", "CLEAN": "clean", "DIRTY": "dirty",
            "EXPENSIVE": "expensive", "CHEAP": "cheap", "NEW": "new", "OLD": "old",
            "YOUNG": "young", "IMPORTANT": "important", "INTERESTING": "interesting",
            "BORING": "boring", "EXCITING": "exciting", "FUNNY": "funny", "SERIOUS": "serious",
            "SMART": "smart", "STUPID": "stupid", "RICH": "rich", "POOR": "poor",
            "HEALTHY": "healthy", "SICK": "sick", "STRONG": "strong", "WEAK": "weak",
            "BUSY": "busy", "FREE": "free", "TIRED": "tired", "ENERGETIC": "energetic"
        }
        
        self.prepositions = {
            "WITH": "with", "WITHOUT": "without", "FOR": "for", "TO": "to",
            "FROM": "from", "IN": "in", "ON": "on", "AT": "at", "BY": "by",
            "NEAR": "near", "FAR": "far", "UNDER": "under", "OVER": "over",
            "BETWEEN": "between", "BEHIND": "behind", "FRONT": "in front of",
            "INSIDE": "inside", "OUTSIDE": "outside", "UP": "up", "DOWN": "down"
        }
        
        self.question_words = {
            "WHO": "who", "WHAT": "what", "WHERE": "where", "WHEN": "when",
            "WHY": "why", "HOW": "how", "WHICH": "which", "HOW-MANY": "how many"
        }

    def generate_simple_sentence(self) -> Tuple[str, str]:
        """Generate a simple gloss-to-English sentence pair."""
        patterns = [
            self._pattern_subject_verb_object,
            self._pattern_time_subject_verb,
            self._pattern_subject_verb_prep_object,
            self._pattern_question,
            self._pattern_descriptive
        ]
        
        pattern = random.choice(patterns)
        return pattern()

    def _pattern_subject_verb_object(self) -> Tuple[str, str]:
        """I EAT APPLE -> I eat an apple."""
        subject = random.choice(list(self.pronouns.keys()))
        verb = random.choice(list(self.verbs.keys()))
        obj = random.choice(list(self.nouns.keys()))
        
        gloss = f"{subject} {verb} {obj}"
        
        # Generate English with proper grammar
        eng_subject = self.pronouns[subject].lower()
        eng_verb = self._conjugate_verb(verb, subject, "present")
        eng_object = self._add_article(self.nouns[obj])
        
        english = f"{eng_subject.capitalize()} {eng_verb} {eng_object}."
        
        return gloss, english

    def _pattern_time_subject_verb(self) -> Tuple[str, str]:
        """YESTERDAY I GO STORE -> Yesterday I went to the store."""
        time = random.choice(list(self.time_markers.keys()))
        subject = random.choice(list(self.pronouns.keys()))
        verb = random.choice(list(self.verbs.keys()))
        obj = random.choice(list(self.nouns.keys()))
        
        gloss = f"{time} {subject} {verb} {obj}"
        
        # Determine tense from time marker
        tense = "past" if time in ["YESTERDAY", "LAST", "BEFORE"] else "present"
        
        eng_time = self.time_markers[time]
        eng_subject = self.pronouns[subject].lower()
        eng_verb = self._conjugate_verb(verb, subject, tense)
        eng_object = self._add_article(self.nouns[obj])
        
        # Add preposition for certain verbs
        prep = "to the" if verb == "GO" else ""
        if prep:
            english = f"{eng_time.capitalize()} {eng_subject} {eng_verb} {prep} {self.nouns[obj]}."
        else:
            english = f"{eng_time.capitalize()} {eng_subject} {eng_verb} {eng_object}."
        
        return gloss, english

    def _pattern_subject_verb_prep_object(self) -> Tuple[str, str]:
        """I WORK WITH FRIEND -> I work with my friend."""
        subject = random.choice(list(self.pronouns.keys()))
        verb = random.choice(list(self.verbs.keys()))
        prep = random.choice(list(self.prepositions.keys()))
        obj = random.choice(list(self.nouns.keys()))
        
        gloss = f"{subject} {verb} {prep} {obj}"
        
        eng_subject = self.pronouns[subject].lower()
        eng_verb = self._conjugate_verb(verb, subject, "present")
        eng_prep = self.prepositions[prep]
        eng_object = self._add_article(self.nouns[obj])
        
        english = f"{eng_subject.capitalize()} {eng_verb} {eng_prep} {eng_object}."
        
        return gloss, english

    def _pattern_question(self) -> Tuple[str, str]:
        """WHERE YOU GO? -> Where do you go?"""
        q_word = random.choice(list(self.question_words.keys()))
        subject = random.choice(["YOU", "HE", "SHE", "THEY"])
        verb = random.choice(list(self.verbs.keys()))
        
        gloss = f"{q_word} {subject} {verb}?"
        
        eng_q_word = self.question_words[q_word]
        eng_subject = self.pronouns[subject].lower()
        
        # Handle auxiliary verb for questions
        if subject == "YOU":
            aux = "do"
        elif subject in ["HE", "SHE"]:
            aux = "does"
        else:
            aux = "do"
        
        eng_verb = self.verbs[verb][0]  # Base form for questions
        
        english = f"{eng_q_word.capitalize()} {aux} {eng_subject} {eng_verb}?"
        
        return gloss, english

    def _pattern_descriptive(self) -> Tuple[str, str]:
        """HOUSE BIG BEAUTIFUL -> The house is big and beautiful."""
        noun = random.choice(list(self.nouns.keys()))
        adj1 = random.choice(list(self.adjectives.keys()))
        adj2 = random.choice(list(self.adjectives.keys()))
        
        gloss = f"{noun} {adj1} {adj2}"
        
        eng_noun = self._add_article(self.nouns[noun])
        eng_adj1 = self.adjectives[adj1]
        eng_adj2 = self.adjectives[adj2]
        
        english = f"{eng_noun.capitalize()} is {eng_adj1} and {eng_adj2}."
        
        return gloss, english

    def _conjugate_verb(self, verb: str, subject: str, tense: str) -> str:
        """Conjugate verb based on subject and tense."""
        verb_forms = self.verbs[verb]
        
        if tense == "past":
            return verb_forms[2]  # Past tense
        elif tense == "present":
            if subject in ["HE", "SHE"]:
                return verb_forms[1]  # Third person singular
            else:
                return verb_forms[0]  # Base form
        else:
            return verb_forms[0]  # Default to base form

    def _add_article(self, noun: str) -> str:
        """Add appropriate article to noun."""
        vowels = 'aeiou'
        if noun[0].lower() in vowels:
            return f"an {noun}"
        else:
            return f"a {noun}"

    def generate_paragraph(self, num_sentences: int = 5) -> Tuple[str, str]:
        """Generate a paragraph with multiple related sentences."""
        # Create thematically related sentences
        theme = random.choice(['daily_routine', 'work_day', 'weekend', 'vacation', 'school'])
        
        if theme == 'daily_routine':
            return self._generate_daily_routine_paragraph(num_sentences)
        elif theme == 'work_day':
            return self._generate_work_day_paragraph(num_sentences)
        elif theme == 'weekend':
            return self._generate_weekend_paragraph(num_sentences)
        elif theme == 'vacation':
            return self._generate_vacation_paragraph(num_sentences)
        else:
            return self._generate_school_paragraph(num_sentences)

    def _generate_daily_routine_paragraph(self, num_sentences: int) -> Tuple[str, str]:
        """Generate daily routine themed paragraph."""
        gloss_sentences = [
            "MORNING I WAKE-UP EARLY",
            "COFFEE I DRINK FIRST", 
            "SHOWER I TAKE QUICK",
            "BREAKFAST I EAT HOME",
            "WORK I GO BY CAR",
            "EVENING I COME HOME TIRED",
            "DINNER I COOK SIMPLE",
            "TV I WATCH RELAX"
        ]
        
        english_sentences = [
            "In the morning I wake up early",
            "I drink coffee first",
            "I take a quick shower", 
            "I eat breakfast at home",
            "I go to work by car",
            "In the evening I come home tired",
            "I cook a simple dinner",
            "I watch TV to relax"
        ]
        
        selected_indices = random.sample(range(len(gloss_sentences)), min(num_sentences, len(gloss_sentences)))
        selected_indices.sort()  # Keep chronological order
        
        gloss_para = ". ".join([gloss_sentences[i] for i in selected_indices]) + "."
        english_para = ". ".join([english_sentences[i] for i in selected_indices]) + "."
        
        return gloss_para, english_para

    def _generate_work_day_paragraph(self, num_sentences: int) -> Tuple[str, str]:
        """Generate work day themed paragraph."""
        gloss_sentences = [
            "YESTERDAY WORK I GO EARLY",
            "MEETING IMPORTANT I HAVE 9 AM",
            "BOSS HAPPY WITH MY PROJECT", 
            "LUNCH I EAT WITH COLLEAGUES",
            "AFTERNOON EMAIL I SEND MANY",
            "5 PM I FINISH WORK",
            "HOME I DRIVE TIRED BUT HAPPY"
        ]
        
        english_sentences = [
            "Yesterday I went to work early",
            "I had an important meeting at 9 AM",
            "My boss was happy with my project",
            "I ate lunch with my colleagues", 
            "In the afternoon I sent many emails",
            "I finished work at 5 PM",
            "I drove home tired but happy"
        ]
        
        selected_indices = random.sample(range(len(gloss_sentences)), min(num_sentences, len(gloss_sentences)))
        selected_indices.sort()
        
        gloss_para = ". ".join([gloss_sentences[i] for i in selected_indices]) + "."
        english_para = ". ".join([english_sentences[i] for i in selected_indices]) + "."
        
        return gloss_para, english_para

    def _generate_weekend_paragraph(self, num_sentences: int) -> Tuple[str, str]:
        """Generate weekend themed paragraph.""" 
        gloss_sentences = [
            "SATURDAY MORNING I SLEEP LATE",
            "COFFEE I MAKE SLOW",
            "FRIEND I CALL PHONE",
            "PARK WE WALK TOGETHER",
            "RESTAURANT WE EAT LUNCH",
            "AFTERNOON MOVIE WE WATCH",
            "EVENING HOME I RELAX"
        ]
        
        english_sentences = [
            "Saturday morning I slept late",
            "I made coffee slowly", 
            "I called my friend on the phone",
            "We walked together in the park",
            "We ate lunch at a restaurant",
            "In the afternoon we watched a movie",
            "In the evening I relaxed at home"
        ]
        
        selected_indices = random.sample(range(len(gloss_sentences)), min(num_sentences, len(gloss_sentences)))
        selected_indices.sort()
        
        gloss_para = ". ".join([gloss_sentences[i] for i in selected_indices]) + "."
        english_para = ". ".join([english_sentences[i] for i in selected_indices]) + "."
        
        return gloss_para, english_para

    def _generate_vacation_paragraph(self, num_sentences: int) -> Tuple[str, str]:
        """Generate vacation themed paragraph."""
        gloss_sentences = [
            "LAST MONTH VACATION I TAKE",
            "BEACH I GO WITH FAMILY",
            "HOTEL BEAUTIFUL NEAR OCEAN",
            "MORNING WE SWIM WATER WARM", 
            "AFTERNOON SUN WE ENJOY",
            "EVENING RESTAURANT WE EAT SEAFOOD",
            "WEEK WONDERFUL WE HAVE"
        ]
        
        english_sentences = [
            "Last month I took a vacation",
            "I went to the beach with my family",
            "The hotel was beautiful near the ocean",
            "In the morning we swam in the warm water",
            "In the afternoon we enjoyed the sun", 
            "In the evening we ate seafood at a restaurant",
            "We had a wonderful week"
        ]
        
        selected_indices = random.sample(range(len(gloss_sentences)), min(num_sentences, len(gloss_sentences)))
        selected_indices.sort()
        
        gloss_para = ". ".join([gloss_sentences[i] for i in selected_indices]) + "."
        english_para = ". ".join([english_sentences[i] for i in selected_indices]) + "."
        
        return gloss_para, english_para

    def _generate_school_paragraph(self, num_sentences: int) -> Tuple[str, str]:
        """Generate school themed paragraph."""
        gloss_sentences = [
            "TODAY SCHOOL I GO EARLY",
            "MATH CLASS I HAVE FIRST",
            "TEACHER EXPLAIN PROBLEM DIFFICULT", 
            "I UNDERSTAND AFTER HELP",
            "LUNCH FRIEND I EAT WITH",
            "AFTERNOON ENGLISH CLASS I ENJOY",
            "HOME I GO HAPPY LEARN"
        ]
        
        english_sentences = [
            "Today I went to school early",
            "I had math class first", 
            "The teacher explained a difficult problem",
            "I understood after getting help",
            "I ate lunch with my friend",
            "In the afternoon I enjoyed English class",
            "I went home happy to learn"
        ]
        
        selected_indices = random.sample(range(len(gloss_sentences)), min(num_sentences, len(gloss_sentences)))
        selected_indices.sort()
        
        gloss_para = ". ".join([gloss_sentences[i] for i in selected_indices]) + "."
        english_para = ". ".join([english_sentences[i] for i in selected_indices]) + "."
        
        return gloss_para, english_para

    def generate_dataset(self, num_sentences: int = 1000, num_paragraphs: int = 200) -> List[Dict]:
        """Generate complete training dataset."""
        dataset = []
        
        # Generate individual sentences
        for _ in range(num_sentences):
            gloss, english = self.generate_simple_sentence()
            dataset.append({
                "input": gloss,
                "target": english,
                "type": "sentence"
            })
        
        # Generate paragraphs
        for _ in range(num_paragraphs):
            num_sents = random.randint(3, 7)  # Paragraphs with 3-7 sentences
            gloss, english = self.generate_paragraph(num_sents)
            dataset.append({
                "input": gloss, 
                "target": english,
                "type": "paragraph"
            })
        
        return dataset

def main():
    """Generate and save the synthetic dataset."""
    print("🔄 Generating ASL Gloss to English synthetic dataset...")
    
    generator = ASLGlossDataGenerator()
    
    # Generate datasets
    train_dataset = generator.generate_dataset(num_sentences=2000, num_paragraphs=400)
    val_dataset = generator.generate_dataset(num_sentences=200, num_paragraphs=50)
    test_dataset = generator.generate_dataset(num_sentences=200, num_paragraphs=50)
    
    # Save datasets
    datasets = {
        "train": train_dataset,
        "validation": val_dataset, 
        "test": test_dataset
    }
    
    with open("gloss_data.json", "w", encoding="utf-8") as f:
        json.dump(datasets, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Dataset generated successfully!")
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"📊 Test samples: {len(test_dataset)}")
    
    # Show some examples
    print("\n📝 Example sentences:")
    for i, example in enumerate(train_dataset[:5]):
        print(f"{i+1}. Gloss: {example['input']}")
        print(f"   English: {example['target']}")
        print()
    
    print("📝 Example paragraph:")
    paragraph_examples = [x for x in train_dataset if x['type'] == 'paragraph'][:1]
    for example in paragraph_examples:
        print(f"Gloss: {example['input']}")
        print(f"English: {example['target']}")

if __name__ == "__main__":
    main()