import cv2
#from matplotlib import pyplot as plt
import numpy as np
#test reading from camera

from detectlines import * 



def annotate_image(image_in):
    """ Given an image Numpy array, return the annotated image as a Numpy array """
    # Only keep white and yellow pixels in the image, all other pixels become black
    image = filter_colors(image_in)

    # Read in and grayscale the image
    #image = (image*255).astype('uint8')  # this step is unnecessary now
    gray = grayscale(image)

    # Apply Gaussian smoothing
    blur_gray = gaussian_blur(gray, kernel_size)

    # Apply Canny Edge Detector
    edges = canny(blur_gray, low_threshold, high_threshold)

    # Create masked edges using trapezoid-shaped region-of-interest
    imshape = image.shape
    vertices = np.array([[\
        ((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]),\
        ((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
        (imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
        (imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0])]]\
        , dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Run Hough on edge detected image
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # Draw lane lines on the original image
    initial_image = image_in.astype('uint8')
    annotated_image = weighted_img(line_image, initial_image)

    return annotated_image




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

def process_road(image):
    return annotate_image(image)



def run_video():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret,frame = cap.read()

        #frame = process_median(frame)

        #frame = process_2gray(frame)

        #frame = process_histogram(frame)

        
        #frame = process_canny(frame)

        #frame = process_sobel(frame)
        frame = process_road(frame)
        

        cv2.imshow('WebCam',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
        
    cap.release()
#test1()
#take_photo()
run_video()
