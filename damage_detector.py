import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import os
from pathlib import Path

class TireDamageDetector:
    def __init__(self):
        # Classification thresholds based on damage count
        self.severity_thresholds = {
            'C1': (1, 5),     # 1-5 chips/cuts
            'C2': (6, 10),    # 6-10 chips/cuts  
            'C3': (11, 15),   # 11-15 chips/cuts
            'C4': (16, 20),   # 16-20 chips/cuts
            'C5': (21, 999)   # 21+ chips/cuts
        }
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return img, blurred
    
    def detect_damage_areas(self, before_img: np.ndarray, after_img: np.ndarray) -> Tuple[np.ndarray, List]:
        """Detect damage by comparing before and after images"""
        
        # Resize images to same size if needed
        h, w = min(before_img.shape[0], after_img.shape[0]), min(before_img.shape[1], after_img.shape[1])
        before_resized = cv2.resize(before_img, (w, h))
        after_resized = cv2.resize(after_img, (w, h))
        
        # Calculate absolute difference
        diff = cv2.absdiff(before_resized, after_resized)
        
        # Threshold to get binary image
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up noise
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours (damage areas)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area (remove very small noise)
        min_area = 50  # Adjust based on your image size
        damage_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        return cleaned, damage_contours
    
    def analyze_damage(self, contours: List, image_shape: Tuple) -> Dict:
        """Analyze detected damage areas"""
        results = {
            'total_damage_area': 0,
            'damage_count': len(contours),
            'damage_details': []
        }
        
        total_image_area = image_shape[0] * image_shape[1]
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Basic shape analysis to distinguish cuts vs chips
            aspect_ratio = max(w, h) / min(w, h)
            damage_type = "cut" if aspect_ratio > 2.5 else "chip"
            
            damage_info = {
                'id': i,
                'type': damage_type,
                'area': area,
                'bbox': (x, y, w, h),
                'aspect_ratio': aspect_ratio
            }
            
            results['damage_details'].append(damage_info)
            results['total_damage_area'] += area
        
        # Store damage count
        damage_count = results['damage_count']
        results['damage_count'] = damage_count
        
        return results
    
    def classify_severity(self, damage_count: int) -> str:
        """Classify damage severity based on count of chips/cuts"""
        if damage_count == 0:
            return 'C0'  # No damage
            
        for severity, (min_count, max_count) in self.severity_thresholds.items():
            if min_count <= damage_count <= max_count:
                return severity
        return 'C5'  # If above 21
    
    def process_tire(self, before_path: str, after_paths: List[str]) -> Dict:
        """Process a single tire with before and multiple after images"""
        
        # Load before image
        before_color, before_gray = self.preprocess_image(before_path)
        
        all_results = []
        
        # Process each after image
        for after_path in after_paths:
            try:
                after_color, after_gray = self.preprocess_image(after_path)
                
                # Detect damage
                damage_mask, contours = self.detect_damage_areas(before_gray, after_gray)
                
                # Analyze damage
                analysis = self.analyze_damage(contours, after_gray.shape)
                
                # Classify severity
                severity = self.classify_severity(analysis['damage_count'])
                
                result = {
                    'after_image': after_path,
                    'damage_analysis': analysis,
                    'severity': severity,
                    'damage_mask': damage_mask
                }
                
                all_results.append(result)
                
            except Exception as e:
                print(f"Error processing {after_path}: {e}")
                continue
        
        # Combine results from all after images (take worst case)
        if all_results:
            worst_severity = self.get_worst_severity([r['severity'] for r in all_results])
            max_damage_count = max([r['damage_analysis']['damage_count'] for r in all_results])
        else:
            worst_severity = 'C0'
            max_damage_count = 0
        
        return {
            'tire_id': Path(before_path).stem.split('_')[1],  # Extract tire number
            'before_image': before_path,
            'individual_results': all_results,
            'final_severity': worst_severity,
            'max_damage_count': max_damage_count
        }
    
    def get_worst_severity(self, severities: List[str]) -> str:
        """Get the worst severity from a list"""
        severity_order = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
        return max(severities, key=lambda x: severity_order.index(x))
    
    def visualize_results(self, result: Dict, save_path: str = None):
        """Visualize detection results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Before image
        before_img = cv2.imread(result['before_image'])
        before_img = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
        axes[0,0].imshow(before_img)
        axes[0,0].set_title('Before')
        axes[0,0].axis('off')
        
        # Show one after image with damage highlighted
        if result['individual_results']:
            best_result = max(result['individual_results'], 
                            key=lambda x: x['damage_analysis']['damage_count'])
            
            after_img = cv2.imread(best_result['after_image'])
            after_img = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)
            
            # Draw bounding boxes on damage
            for damage in best_result['damage_analysis']['damage_details']:
                x, y, w, h = damage['bbox']
                cv2.rectangle(after_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(after_img, damage['type'], (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            axes[0,1].imshow(after_img)
            axes[0,1].set_title(f'After - {result["final_severity"]}')
            axes[0,1].axis('off')
            
            # Damage mask
            axes[1,0].imshow(best_result['damage_mask'], cmap='gray')
            axes[1,0].set_title('Detected Damage')
            axes[1,0].axis('off')
            
            # Summary text
            summary_text = f"""
Tire ID: {result['tire_id']}
Final Severity: {result['final_severity']}
Max Damage Count: {result['max_damage_count']}
Damage Count: {best_result['damage_analysis']['damage_count']}
            """
            axes[1,1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
            axes[1,1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Results saved to: {save_path}")
        plt.show()