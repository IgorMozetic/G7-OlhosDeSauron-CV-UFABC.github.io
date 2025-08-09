import cv2
import numpy as np

def ImagePreProcessing(image: np.ndarray) -> np.ndarray:
   
    imageBlurred = cv2.GaussianBlur(image, (5, 5), 3)
   
    imageBorders = cv2.Canny(imageBlurred, 90, 140)
  
    kernel = np.ones((4, 4), np.uint8)

    imageDilated = cv2.dilate(imageBorders, kernel, iterations=2)
    
    imageEroded = cv2.erode(imageDilated, kernel, iterations=1)
    
    imagePreProcessed = imageEroded.copy()

    return imagePreProcessed
