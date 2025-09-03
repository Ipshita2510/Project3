#!/usr/bin/env python3
"""
Quick test script for tire damage detection
"""

import os
import glob
from damage_detector import TireDamageDetector

def test_single_tire():
    """Test with a single tire"""
    detector = TireDamageDetector()
    
    # Update these paths to match your images
    before_path = "data/before/tire_001_before_1.jpg"
    after_paths = [
        "data/after/tire_001_after_1.jpg",
        "data/after/tire_001_after_2.jpg",
    ]
   
    # Check if files exist
    if not os.path.exists(before_path):
        print(f"Before image not found: {before_path}")
        return
    
    existing_after = [p for p in after_paths if os.path.exists(p)]
    if not existing_after:
        print("No after images found!")
        return
    
    print(f"Processing tire with {len(existing_after)} after images...")
    
    try:
        result = detector.process_tire(before_path, existing_after)
        
        print("\n" + "="*50)
        print(f"RESULT: Tire {result['tire_id']}")
        print(f"Final Severity: {result['final_severity']}")
        print(f"Max Damage Count: {result['max_damage_count']}")
        print("="*50)
        
        # Show detailed results for each after image
        for i, res in enumerate(result['individual_results']):
            print(f"\nAfter Image {i+1}:")
            print(f"  - Damage count: {res['damage_analysis']['damage_count']}")
            print(f"  - Severity: {res['severity']}")
            for j, damage in enumerate(res['damage_analysis']['damage_details']):
                print(f"    Damage {j+1}: {damage['type']} (area: {damage['area']:.0f}px)")
        
        # Visualize results
        detector.visualize_results(result, f"results/tire_{result['tire_id']}_analysis.png")
        
    except Exception as e:
        print(f"Error: {e}")

def test_all_tires():
    """Test all tires in your dataset"""
    detector = TireDamageDetector()
    
    # Get all before images with your naming convention
    before_images = glob.glob("data/before/tire_*_before_*.jpg") + glob.glob("data/before/tire_*_before_*.png")
    
    results = []
    
    for before_path in before_images:
        # Extract tire ID from filename (e.g., tire_001_before_1.jpg -> 001)
        filename = os.path.basename(before_path)
        tire_id = filename.split('_')[1]  # Gets "001" from "tire_001_before_1.jpg"
        
        # Find corresponding after images
        after_paths = []
        for i in range(1, 8):  # Check for up to 7 after images
            after_path = f"data/after/tire_{tire_id}_after_{i}.jpg"
            if os.path.exists(after_path):
                after_paths.append(after_path)
        
        if not after_paths:
            print(f"No after images found for tire {tire_id}")
            continue
        
        try:
            result = detector.process_tire(before_path, after_paths)
            results.append(result)
            print(f"✓ Tire {tire_id}: {result['final_severity']} ({result['max_damage_count']} damages)")
            
        except Exception as e:
            print(f"✗ Error processing tire {tire_id}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: Processed {len(results)} tires")
    
    severity_counts = {}
    for result in results:
        sev = result['final_severity']
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    print("Severity Distribution:")
    for sev in ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']:
        count = severity_counts.get(sev, 0)
        print(f"  {sev}: {count} tires")
    
    return results

if __name__ == "__main__":
    print("Tire Damage Detection - Test Script")
    print("1. Testing single tire")
    print("2. Testing all tires")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        test_single_tire()
    elif choice == "2":
        test_all_tires()
    else:
        print("Invalid choice. Running single tire test...")
        test_single_tire()