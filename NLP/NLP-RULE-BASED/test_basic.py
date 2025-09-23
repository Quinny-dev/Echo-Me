#!/usr/bin/env python3
"""
Quick validation script to test if the ASL translation pipeline works correctly.
Run this after installation to verify everything is working.
"""

def test_rule_based():
    """Test the rule-based translator."""
    print("Testing rule-based translator...")
    
    try:
        from rule_based import RuleBasedTranslator
        translator = RuleBasedTranslator()
        
        # Test basic translation
        result = translator.translate("ME WANT GO STORE")
        assert result['rough_english'] == "I want go store."
        assert result['detected_tense'] == "present"
        print("✅ Rule-based translator works!")
        return True
        
    except Exception as e:
        print(f"❌ Rule-based translator failed: {e}")
        return False

def test_grammar_corrector():
    """Test the grammar corrector."""
    print("Testing grammar corrector (this may take a moment)...")
    
    try:
        from corrector import GrammarCorrector
        corrector = GrammarCorrector()
        
        # Test basic correction
        result = corrector.correct_single("I want go store.")
        corrected = result['corrected']
        
        # Should be grammatically better
        assert len(corrected) > 10  # Should produce some output
        assert corrected.endswith('.')  # Should end with punctuation
        print(f"✅ Grammar corrector works! Output: {corrected}")
        return True
        
    except Exception as e:
        print(f"❌ Grammar corrector failed: {e}")
        return False

def test_pipeline():
    """Test the complete pipeline."""
    print("Testing complete pipeline...")
    
    try:
        from demo import ASLTranslationPipeline
        pipeline = ASLTranslationPipeline()
        
        # Test translation
        result = pipeline.translate("ME WANT EAT PIZZA", show_steps=False)
        
        assert 'final_output' in result
        assert len(result['final_output']) > 5
        print(f"✅ Pipeline works! Output: {result['final_output']}")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return False

def main():
    """Run all tests."""
    print("ASL Translation Pipeline - Quick Validation")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test each component
    if test_rule_based():
        tests_passed += 1
    
    print()
    
    if test_grammar_corrector():
        tests_passed += 1
    
    print()
    
    if test_pipeline():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All systems working! You can now use the translator.")
        print("\nTry running: python demo.py")
    else:
        print("⚠️  Some components failed. Check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that you have internet connection (for first-time model download)")
        print("3. Ensure you have at least 2GB free RAM")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)