import cv2
import numpy as np

# Read images as color
img = cv2.imread('pt-3.0-circles.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('pt-3.0-circles2.png', cv2.IMREAD_COLOR)
img3 = cv2.imread('pt-3.0-circles3.png', cv2.IMREAD_COLOR)

# Convert to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# Blur the images to reduce noise
img_blur = cv2.medianBlur(gray, 5)
img_blur2 = cv2.medianBlur(gray2, 5)
img_blur3 = cv2.medianBlur(gray3, 5)

# Apply Hough Transform to detect circles in the images
circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 40, param1=50, param2=40, minRadius=10, maxRadius=40)
circles2 = cv2.HoughCircles(img_blur2, cv2.HOUGH_GRADIENT, 1, 40, param1=50, param2=40, minRadius=10, maxRadius=50)
circles3 = cv2.HoughCircles(img_blur3, cv2.HOUGH_GRADIENT, 1, 40, param1=50, param2=40, minRadius=10, maxRadius=50)

# Draw detected circles on the first image
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle (center)
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

# Draw detected circles on the second image
if circles2 is not None:
    circles2 = np.uint16(np.around(circles2))
    for i in circles2[0, :]:
        # Draw outer circle
        cv2.circle(img2, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle (center)
        cv2.circle(img2, (i[0], i[1]), 2, (0, 0, 255), 3)

# Draw detected circles on the third image
if circles3 is not None:
    circles3 = np.uint16(np.around(circles3))
    for i in circles3[0, :]:
        # Draw outer circle
        cv2.circle(img3, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle (center)
        cv2.circle(img3, (i[0], i[1]), 2, (0, 0, 255), 3)

# Show the results
cv2.imshow("Circulos detectados imagem 1", img)
cv2.imshow("Circulos detectados imagem 2", img2)
cv2.imshow("Circulos detectados imagem 3", img3)

cv2.waitKey(0)
cv2.destroyAllWindows()

