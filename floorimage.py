import cv2
import numpy as np
from PIL import Image

def convert_3d_image_to_floor_plan(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection using Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty image for the floor plan
    floor_plan = np.zeros_like(gray)
    
    # Draw contours on the floor plan
    cv2.drawContours(floor_plan, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    
    # Convert the floor plan to an image
    floor_plan_image = Image.fromarray(floor_plan)
    
    return floor_plan_image

# Example usage
image_path = 'pexels-rostislav-5011647.jpg'
floor_plan_image = convert_3d_image_to_floor_plan(image_path)
floor_plan_image.show()
