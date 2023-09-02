import cv2
#from matplotlib import pyplot as plt
import numpy as np
#test reading from camera



def test1():
    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
    print(ret)
    print(frame)
    cap.release()

def take_photo():
    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
    cv2.imwrite('photo2.jpg',frame)
    cap.release()

def process_median(image):
    return cv2.medianBlur(image,3)

def process_2gray(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

def process_histogram(image):
    return cv2.equalizeHist(image)

def process_sobel(image):
    # Apply Sobel operator in X direction
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

    # Apply Sobel operator in Y direction
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # You can also compute the magnitude of the gradient
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    return gradient_magnitude



def process_canny(image):
    return cv2.Canny(image,50,150)

def run_video():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret,frame = cap.read()

        frame = process_median(frame)

        frame = process_2gray(frame)

        #frame = process_histogram(frame)

        
        #frame = process_canny(frame)

        frame = process_sobel(frame)

        cv2.imshow('WebCam',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
        
    cap.release()
#test1()
#take_photo()
run_video()
