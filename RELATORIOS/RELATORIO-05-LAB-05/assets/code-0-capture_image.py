import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Display the resulting frame
    cv.imshow('frame', frame)

    k = cv.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('s'): # wait for 's' key to save
      cv.imwrite('item.png',frame)

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
