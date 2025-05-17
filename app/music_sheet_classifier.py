import os
import string
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Union
import pytesseract

class MusicSheetClassifier:
    VALID_CHARS = set("12345670.-|/\[] ")
    INVALID_CHARS = set("89") | set(string.ascii_lowercase) | set(string.ascii_uppercase)
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    """
    A classifier for different music notation systems including:
    - Staff notation (Western standard notation)
    - Jianpu (Numbered musical notation)`
    - Tablature
    - Other notation systems
    """
    
    def __init__(self):
        """Initialize the classifier with default parameters"""
        # Thresholds for classification
        self.staff_line_threshold = 0.6  # Minimum ratio of horizontal lines to consider staff notation
        self.jianpu_number_threshold = 0.4  # Minimum ratio of numeric characters to consider jianpu
        self.tablature_threshold = 0.5  # Minimum ratio of string-like features to consider tablature
        
        # Common preprocessing parameters
        self.resize_width = 1000  # Width to resize images to
        
    def preprocess_image(self, image_path: Union[str, np.ndarray]) -> np.ndarray:
        """
        Preprocess the image for feature extraction
        
        Args:
            image_path: Path to the image or numpy array of the image
            
        Returns:
            Preprocessed image as numpy array
        """
        # Load image if path is provided
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Could not load image from {image_path}")
        else:
            image = image_path.copy()
            
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Resize for consistent processing
        height, width = gray.shape
        new_height = int(height * (self.resize_width / width))
        resized = cv2.resize(gray, (self.resize_width, new_height))
        
        # Binarize image
        _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return binary
    
    def detect_staff_notation(self, binary_img: np.ndarray) -> Tuple[bool, float]:
        """
        Simplified function to detect if the image contains standard staff notation
        by checking for the presence of staff lines.
        
        Args:
            binary_img: Preprocessed binary image
            
        Returns:
            Tuple of (is_staff_notation, confidence)
        """
        # Use morphological operations to detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_detected = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Find contours of horizontal lines
        contours, _ = cv2.findContours(horizontal_detected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Store information about detected horizontal lines
        horizontal_lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Staff lines are thin and extend across most of the width
            if h <= 3 and w > binary_img.shape[1] // 3:
                horizontal_lines.append((y, h))  # Store y-position and height
        
        # Sort lines by y-position
        horizontal_lines.sort(key=lambda line: line[0])
        
        # Need at least 5 horizontal lines for a standard staff
        if len(horizontal_lines) < 5:
            return False, 0.0
        
        # Check for groups of evenly spaced lines (staff lines)
        staff_groups = []
        current_group = [horizontal_lines[0][0]]  # Start with first line
        
        for i in range(1, len(horizontal_lines)):
            current_y = horizontal_lines[i][0]
            prev_y = horizontal_lines[i-1][0]
            spacing = current_y - prev_y
            
            # If spacing is relatively consistent, add to current group
            if spacing < binary_img.shape[0] * 0.05:  # 5% of image height as threshold
                current_group.append(current_y)
            else:
                # If we have at least 5 lines, consider it a staff
                if len(current_group) >= 5:
                    staff_groups.append(current_group)
                # Start a new group
                current_group = [current_y]
        
        # Check the last group
        if len(current_group) >= 5:
            staff_groups.append(current_group)
        
        # Calculate confidence based on number of staff groups found
        if len(staff_groups) > 0:
            # Higher confidence with more staff groups
            confidence = min(1.0, len(staff_groups) * 0.5)
            return True, confidence
        else:
            return False, 0.0
    
    
    def detect_jianpu_notation(self, binary_img: np.ndarray) -> Tuple[bool, float]:
        """
        Detect Jianpu notation from a binary image using OCR and layout analysis.

        Args:
            binary_img: Preprocessed binary image (black text on white background)

        Returns:
            Tuple (is_jianpu, confidence)
        """
        # OCR with layout info
        ocr_data = pytesseract.image_to_data(binary_img, config='--oem 3 --psm 6', output_type=pytesseract.Output.DICT)

        lines = {}
        n = len(ocr_data['text'])

        for i in range(n):
            text = ocr_data['text'][i].strip()
            line_num = ocr_data['line_num'][i]
            top = ocr_data['top'][i]
            height = ocr_data['height'][i]

            if not text:
                continue

            if line_num not in lines:
                lines[line_num] = {
                    "text": [],
                    "tops": [],
                    "heights": []
                }

            lines[line_num]["text"].append(text)
            lines[line_num]["tops"].append(top)
            lines[line_num]["heights"].append(height)


        char_ratio_sum = 0.0
        valid_line_count = 0
        total_line_count = 0
        # jianpu_lines = []

        for line_info in lines.values():
            line_text = ' '.join(line_info["text"])
            # Skip lines that clearly do not belong to Jianpu
            if any(c.isalpha() for c in line_text) or '8' in line_text or '9' in line_text:
                continue



            chars = [c for c in line_text if not c.isspace()]
            total_chars = len(chars)
            if total_chars < 5:
                continue

            if (sum(1 for c in chars if c in self.INVALID_CHARS) > 0):
                continue

            valid_chars = sum(1 for c in chars if c in self.VALID_CHARS)
            char_ratio = valid_chars / total_chars
            # if (char_ratio != 1):
            #     print(f"chars({chars})")
            #     print(f"char_ratio({char_ratio}) = valid_chars({valid_chars}) / total_chars({total_chars})")
            total_line_count += 1
            char_ratio_sum += char_ratio

            # Heuristic: at least 90% valid characters
            if char_ratio >= 0.9:
                valid_line_count += 1
                # char_ratio_sum += char_ratio
                # avg_top = sum(line_info["tops"]) / len(line_info["tops"])
                # avg_height = sum(line_info["heights"]) / len(line_info["heights"])
                # jianpu_lines.append((avg_top, avg_height))

        if total_line_count == 0:
            return False, 0.0
        
        

        # Check vertical spacing consistency
        # jianpu_lines.sort()
        # spacing_diffs = [
        #     jianpu_lines[i+1][0] - jianpu_lines[i][0]
        #     for i in range(len(jianpu_lines) - 1)
        # ]
        # consistent_spacing = 0
        # if spacing_diffs:
        #     avg_spacing = sum(spacing_diffs) / len(spacing_diffs)
        #     consistent_spacing = sum(
        #         0.8 <= diff / avg_spacing <= 1.2 for diff in spacing_diffs
        #     ) / len(spacing_diffs)

        # Final confidence: content match * layout consistency
        
        content_score = char_ratio_sum / total_line_count
        # content_score = char_ratio_sum / valid_line_count
        # if valid_line_count > 0:
        #     content_score = char_ratio_sum / valid_line_count
        # else:
        #     content_score = 0.0
        # layout_score = consistent_spacing if spacing_diffs else 0.0
        # if layout_score == 0:
        #     print(f"consistent_spacing:({consistent_spacing})spacing_diffs:({spacing_diffs})jianpu_lines({jianpu_lines})")
        # confidence = 0.8 * content_score + 0.2 * layout_score
        confidence = content_score 


        return confidence > 0.6, confidence
    
   
   
    def detect_tablature(self, binary_img: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if the image contains tablature notation (guitar/bass tabs)
        
        Args:
            binary_img: Preprocessed binary image
            
        Returns:
            Tuple of (is_tablature, confidence)
        """
        # Tablature has equally spaced horizontal lines with numbers on them
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detected_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Count horizontal lines
        contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get y-positions of lines
        line_positions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > binary_img.shape[1] // 3:  # Long enough to be a tab line
                line_positions.append(y + h//2)
        
        # Sort line positions
        line_positions.sort()
        
        # Check if lines are evenly spaced (a characteristic of tablature)
        even_spacing = False
        if len(line_positions) >= 4:  # At least 4 strings
            spacing = []
            for i in range(1, len(line_positions)):
                spacing.append(line_positions[i] - line_positions[i-1])
            
            # Calculate standard deviation of spacing
            if spacing:
                mean_spacing = sum(spacing) / len(spacing)
                std_spacing = sum((s - mean_spacing) ** 2 for s in spacing) ** 0.5
                
                # Evenly spaced if standard deviation is low relative to mean
                even_spacing = std_spacing / mean_spacing < 0.2 if mean_spacing > 0 else False
        
        # Look for digit-like contours between the lines
        digit_between_lines = 0
        total_between_lines = 0
        
        if len(line_positions) >= 2:
            min_y = min(line_positions)
            max_y = max(line_positions)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                center_y = y + h//2
                
                # Check if contour is between the lines
                if min_y <= center_y <= max_y:
                    total_between_lines += 1
                    
                    # Check if it's digit-like (small and somewhat square)
                    aspect_ratio = h / w if w > 0 else 0
                    if 0.5 <= aspect_ratio <= 2.0 and 5 < w < 30 and 5 < h < 30:
                        digit_between_lines += 1
        
        # Calculate confidence
        digit_line_ratio = digit_between_lines / max(1, total_between_lines)
        tab_confidence = 0.5 * even_spacing + 0.5 * digit_line_ratio if len(line_positions) >= 4 else 0.0
        
        is_tablature = tab_confidence > self.tablature_threshold
        
        return is_tablature, tab_confidence
    
    def classify(self, image_path: Union[str, np.ndarray]) -> Dict:
        """
        Classify the type of music notation in the image
        
        Args:
            image_path: Path to the image or numpy array of the image
            
        Returns:
            Dictionary with classification results including:
            - detected_type: The most likely notation type
            - confidences: Confidence scores for each notation type
            - features: Extracted features used for classification
        """
        # Preprocess image
        binary_img = self.preprocess_image(image_path)
        
        # Detect different notation types
        is_staff, staff_confidence = self.detect_staff_notation(binary_img)
        is_jianpu, jianpu_confidence = self.detect_jianpu_notation(binary_img)
        is_tablature, tablature_confidence = self.detect_tablature(binary_img)
        
        # Compile confidences
        confidences = {
            "staff_notation": float(staff_confidence),
            "jianpu_notation": float(jianpu_confidence),
            "tablature": float(tablature_confidence),
        }
        
        # Determine the most likely notation type
        max_confidence_type = max(confidences, key=confidences.get)
        max_confidence = confidences[max_confidence_type]
        
        # Only classify if we have reasonable confidence
        if max_confidence < 0.3:
            detected_type = "unknown"
        else:
            detected_type = max_confidence_type
        
        # Construct result
        result = {
            "detected_type": detected_type,
            "confidences": confidences,
            "features": {
                "staff_lines_detected": is_staff,
                "jianpu_numbers_detected": is_jianpu,
                "tablature_structure_detected": is_tablature
            }
        }
        
        return result

    def explain_classification(self, image_path: Union[str, np.ndarray], 
                              classification_result: Dict = None) -> Dict:
        """
        Provide a visual explanation of the classification process
        
        Args:
            image_path: Path to the image or numpy array of the image
            classification_result: Optional pre-computed classification result
            
        Returns:
            Dictionary with visualization data
        """
        # Load image if path is provided
        if isinstance(image_path, str):
            original_img = cv2.imread(image_path)
            if original_img is None:
                raise FileNotFoundError(f"Could not load image from {image_path}")
        else:
            original_img = image_path.copy()
            
        # Convert to RGB for visualization
        if len(original_img.shape) == 3:
            rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
            
        # Preprocess image
        binary_img = self.preprocess_image(image_path)
        
        # Classify if not provided
        if classification_result is None:
            classification_result = self.classify(image_path)
            
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Original image
        axes[0, 0].imshow(rgb_img)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Preprocessed image
        axes[0, 1].imshow(binary_img, cmap='gray')
        axes[0, 1].set_title("Preprocessed Image")
        axes[0, 1].axis('off')
        
        # Feature visualization (simplified for this example)
        feature_img = cv2.cvtColor(binary_img.copy(), cv2.COLOR_GRAY2RGB)
        
        # If staff notation, highlight horizontal lines
        if classification_result["features"]["staff_lines_detected"]:
            lines = cv2.HoughLinesP(binary_img, 1, np.pi/180, threshold=100, 
                                  minLineLength=binary_img.shape[1] // 3, maxLineGap=20)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(y2 - y1) < 10:  # Nearly horizontal line
                        cv2.line(feature_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        axes[1, 0].imshow(feature_img)
        axes[1, 0].set_title("Detected Features")
        axes[1, 0].axis('off')
        
        # Classification result
        detected_type = classification_result["detected_type"].replace("_", " ").title()
        confidence_values = classification_result["confidences"]
        
        # Sort confidences in descending order
        sorted_confidences = sorted(confidence_values.items(), key=lambda x: x[1], reverse=True)
        
        # Plot confidences as bar chart
        labels = [k.replace("_", " ").title() for k, v in sorted_confidences]
        values = [v for k, v in sorted_confidences]
        
        axes[1, 1].bar(labels, values, color='skyblue')
        axes[1, 1].set_title(f"Classification Result: {detected_type}")
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_ylabel("Confidence")
        
        # Rotate labels for better visibility
        plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        return {
            "classification": classification_result,
            "figure": fig
        }

# Usage example
def main():
    # Initialize classifier
    classifier = MusicSheetClassifier()
    
    # Specify the directory where your images are located
    image_directory = "test_images"
    
    # Loop through all files in the directory
    for filename in os.listdir(image_directory):
        # Only process image files (assuming .jpg for this case)
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(image_directory, filename)
            result = classifier.classify(image_path)
            
            print(f"({filename}): {result['detected_type']} {result['confidences']}")

    # explanation = classifier.explain_classification("test_images/o_05.png")
    # plt.show()

if __name__ == "__main__":
    main()