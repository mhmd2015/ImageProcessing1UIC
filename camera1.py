import cv2
#from matplotlib import pyplot as plt

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
    
def run_video():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret,frame = cap.read()
        cv2.imshow('WebCam',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
        
    cap.release()
#test1()
#take_photo()
run_video()
