import numpy as np
import cv2

# Set up specifically for the ELP 1.0 Megapixel Dual Lens Usb Camera Module ...
# https://www.amazon.com/ELP-Megapixel-Camera-Module-Biometric/dp/B00VG32EC2/ref=sr_1_4?ie=UTF8&qid=1502756603&sr=8-4&keywords=stereo+camera
# This camera enumerates as two separate cameras, hence the separate `left` and
# `right` VideoCapture instances. For a proper stereo camera with a common
# clock, use one VideoCapture instance and pass in whether you want the 0th or
# 1st camera in retrieve().

# TODO: Use more stable identifiers
left = cv2.VideoCapture(0)
right = cv2.VideoCapture(1)

# Increase the resolution
left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# Grab both frames first, then retrieve to minimize latency between cameras
while(left.grab() and right.grab()):
    _, leftFrame = left.retrieve()
    leftWidth, leftHeight = leftFrame.shape[:2]
    _, rightFrame = right.retrieve()
    rightWidth, rightHeight = rightFrame.shape[:2]

    # TODO: Calibrate the cameras and correct the images

    cv2.imshow('left', leftFrame)
    cv2.imshow('right', rightFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

left.release()
right.release()
cv2.destroyAllWindows()