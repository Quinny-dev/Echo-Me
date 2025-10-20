"""
Demo script showcasing ASL Gloss to English translation capabilities.
Demonstrates single sentences, paragraphs, and batch processing.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import ASLTranslator
import time

def demo_single_sentences(translator: ASLTranslator):
    """Demonstrate single sentence translations."""
    print("🎯 Single Sentence Translation Demo")
    print("=" * 50)
    
    test_sentences = [
        "YESTERDAY I GO STORE WITH FRIEND",
        "MORNING COFFEE I DRINK HOT",
        "BOOK INTERESTING I READ LAST WEEK",
        "TOMORROW WORK I START EARLY",
        "FAMILY DINNER WE EAT TOGETHER",
        "WEATHER TODAY BEAUTIFUL SUNNY",
        "MOVIE LAST NIGHT WE WATCH FUNNY",
        "SCHOOL I STUDY MATH DIFFICULT",
        "DOCTOR APPOINTMENT I HAVE FRIDAY",
        "BIRTHDAY PARTY FRIEND I GO SATURDAY"
    ]
    
    for i, gloss in enumerate(test_sentences, 1):
        start_time = time.time()
        translation = translator.translate(gloss)
        end_time = time.time()
        
        print(f"\n{i}. Gloss: {gloss}")
        print(f"   English: {translation}")
        print(f"   Time: {(end_time - start_time)*1000:.0f}ms")

def demo_paragraph_translation(translator: ASLTranslator):
    """Demonstrate paragraph-level translations."""
    print("\n\n📖 Paragraph Translation Demo")
    print("=" * 50)
    
    paragraphs = [
        {
            "title": "Daily Routine",
            "gloss": "YESTERDAY MORNING I WAKE-UP EARLY. COFFEE I DRINK FIRST. SHOWER I TAKE QUICK. BREAKFAST I EAT HOME. WORK I GO BY CAR. TRAFFIC BAD TODAY. OFFICE I ARRIVE 9 AM. MEETING IMPORTANT I HAVE WITH BOSS. PROJECT NEW WE DISCUSS. LUNCH I EAT WITH COLLEAGUES RESTAURANT. AFTERNOON EMAIL I SEND MANY. 5 PM I FINISH WORK. HOME I DRIVE TIRED BUT HAPPY."
        },
        {
            "title": "Weekend Activities", 
            "gloss": "SATURDAY MORNING I SLEEP LATE. 10 AM I WAKE-UP RELAX. COFFEE I MAKE SLOW. FRIEND I CALL PHONE CHAT. PARK WE DECIDE GO WALK. WEATHER BEAUTIFUL PERFECT. DOG MANY WE SEE PLAY. LAKE WE SIT BENCH TALK. LUNCH RESTAURANT WE EAT PIZZA. AFTERNOON MOVIE WE WATCH THEATER. COMEDY FILM VERY FUNNY. EVENING HOME I COOK DINNER SIMPLE. TV I WATCH NEWS RELAX."
        },
        {
            "title": "School Experience",
            "gloss": "TODAY SCHOOL I GO EARLY BUS. MATH CLASS I HAVE FIRST PERIOD. TEACHER EXPLAIN ALGEBRA PROBLEM DIFFICULT. I UNDERSTAND AFTER HELP ASK. FRIEND STUDY GROUP WE MAKE. LIBRARY WE GO WORK TOGETHER. ENGLISH CLASS AFTERNOON I ENJOY. SHAKESPEARE PLAY WE READ. HOMEWORK MUCH I HAVE TONIGHT. HOME I GO STUDY PREPARE EXAM TOMORROW."
        }
    ]
    
    for i, para in enumerate(paragraphs, 1):
        print(f"\n{i}. {para['title']}")
        print(f"Gloss: {para['gloss']}")
        print()
        
        start_time = time.time()
        translation = translator.translate_paragraph(para['gloss'])
        end_time = time.time()
        
        print(f"English: {translation}")
        print(f"Time: {(end_time - start_time)*1000:.0f}ms")

def demo_batch_processing(translator: ASLTranslator):
    """Demonstrate batch processing capabilities."""
    print("\n\n⚡ Batch Processing Demo")
    print("=" * 50)
    
    batch_inputs = [
        "I EAT BREAKFAST MORNING",
        "YOU GO SCHOOL BUS",
        "HE PLAY SOCCER FIELD", 
        "SHE READ BOOK LIBRARY",
        "WE WATCH MOVIE THEATER",
        "THEY VISIT MUSEUM WEEKEND",
        "RAIN TODAY HEAVY",
        "SUN SHINE BRIGHT YESTERDAY",
        "WIND COLD WINTER",
        "FLOWER BLOOM SPRING BEAUTIFUL"
    ]
    
    print(f"Processing {len(batch_inputs)} sentences in batch...")
    
    start_time = time.time()
    translations = translator.translate_batch(batch_inputs)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = (total_time / len(batch_inputs)) * 1000
    
    print(f"\nResults:")
    for i, (gloss, translation) in enumerate(zip(batch_inputs, translations), 1):
        print(f"{i:2d}. {gloss:<30} → {translation}")
    
    print(f"\nBatch Processing Stats:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average per sentence: {avg_time:.0f}ms")
    print(f"Throughput: {len(batch_inputs)/total_time:.1f} sentences/second")

def demo_question_translation(translator: ASLTranslator):
    """Demonstrate question translation."""
    print("\n\n❓ Question Translation Demo")
    print("=" * 50)
    
    questions = [
        "WHERE YOU GO TOMORROW?",
        "WHAT YOU EAT BREAKFAST?",
        "WHO YOUR FRIEND NAME?",
        "WHEN MEETING START?",
        "WHY YOU STUDY ENGLISH?",
        "HOW YOU GO WORK?",
        "WHICH BOOK YOU LIKE BEST?",
        "HOW-MANY SISTER YOU HAVE?"
    ]
    
    for i, question in enumerate(questions, 1):
        translation = translator.translate(question)
        print(f"{i}. {question:<30} → {translation}")

def demo_comparative_analysis(translator: ASLTranslator):
    """Demonstrate different beam search settings."""
    print("\n\n🔍 Comparative Analysis Demo")
    print("=" * 50)
    
    test_gloss = "YESTERDAY BEAUTIFUL DAY I FRIEND GO PARK WALK DOG PLAY HAPPY"
    
    settings = [
        {"num_beams": 1, "do_sample": False, "name": "Greedy"},
        {"num_beams": 2, "do_sample": False, "name": "Beam-2"},
        {"num_beams": 4, "do_sample": False, "name": "Beam-4"},
        {"num_beams": 1, "do_sample": True, "name": "Sampling"}
    ]
    
    print(f"Input Gloss: {test_gloss}\n")
    
    for setting in settings:
        start_time = time.time()
        translation = translator.translate(
            test_gloss, 
            num_beams=setting["num_beams"],
            do_sample=setting["do_sample"]
        )
        end_time = time.time()
        
        print(f"{setting['name']:12} → {translation}")
        print(f"{'':12}   Time: {(end_time - start_time)*1000:.0f}ms")

def print_model_info(translator: ASLTranslator):
    """Print model information."""
    print("🤖 Model Information")
    print("=" * 50)
    print(f"Model Path: {translator.model_path}")
    print(f"Device: {translator.device}")
    print(f"Tokenizer Vocab Size: {len(translator.tokenizer)}")
    print(f"Model Type: {translator.model.__class__.__name__}")
    
    # Calculate model size
    param_count = sum(p.numel() for p in translator.model.parameters())
    size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32
    
    print(f"Parameters: {param_count:,}")
    print(f"Estimated Size: {size_mb:.1f} MB")

def main():
    """Main demo function."""
    # Model path - adjust as needed
    model_path = "../models/distilt5-asl-finetuned"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("💡 Please train the model first using: python ../src/train.py")
        return
    
    print("🚀 ASL Gloss to English Translation System Demo")
    print("=" * 60)
    
    try:
        # Initialize translator
        print("📥 Loading model...")
        translator = ASLTranslator(model_path)
        print("✅ Model loaded successfully!\n")
        
        # Print model information
        print_model_info(translator)
        
        # Run demos
        demo_single_sentences(translator)
        demo_paragraph_translation(translator)
        demo_batch_processing(translator)
        demo_question_translation(translator)
        demo_comparative_analysis(translator)
        
        print("\n\n🎉 Demo completed successfully!")
        print("\nTo try interactive mode, run:")
        print("python ../src/inference.py --interactive")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()