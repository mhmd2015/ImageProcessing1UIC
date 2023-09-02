import cv2
#from matplotlib import pyplot as plt

#test reading from camera

CAM_RIGHT = 1
CAM_LEFT = 0

def test1(prt=0):
    cap = cv2.VideoCapture(prt)
    ret,frame = cap.read()
    print(ret)
    #print(frame)
    cap.release()

   

def show_photo(prt):
    cap = cv2.VideoCapture(prt)
    ret,frame = cap.read()
    #cv2.imwrite('photo2.jpg',frame)
    
    
    winname = "Test"+str(prt)
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, 40+prt*300,30)  # Move it to (40,30)
    cv2.imshow(winname, frame)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cap.release()

def take_photo(prt=0):
    cap = cv2.VideoCapture(prt)
    ret, frame = cap.read()
    cap.release()
    return frame

def show_photo_stereo():
    imgL = take_photo(CAM_LEFT)
    imgR = take_photo(CAM_RIGHT)

    stereo = cv2.StereoBM.create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL,imgR)

    winname = "Test"
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
    cv2.imshow(winname, disparity)
    cv2.waitKey()
    cv2.destroyAllWindows()


    
def run_video(prt=0):
    cap = cv2.VideoCapture(prt)
    while cap.isOpened():
        ret,frame = cap.read()
        cv2.imshow('WebCam',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
        
    cap.release()
#test1(CAM_RIGHT)
#test1(CAM_LEFT)
#
#take_photo(CAM_RIGHT)
#take_photo(CAM_LEFT)

#run_video(CAM_LEFT)
#run_video(CAM_RIGHT)

show_photo_stereo()
