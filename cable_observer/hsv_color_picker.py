import os
import glob
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="HSV color picker")
parser.add_argument('-d', '--dir', type=str, default="test", help='Dataset directory')
args = parser.parse_args()
path = os.getcwd()
path = os.path.abspath(os.path.join(path, os.pardir)) + "/media/"

def nothing(x):
    pass

images_paths = sorted(glob.glob(path + "*.jpg"))
idx = 0
img = cv2.imread(images_paths[idx])
output = img
waitTime = 33

print(images_paths)
# Create a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)  # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
cv2.createTrackbar('VMax', 'image', 0, 255, nothing)
cv2.createTrackbar("Idx", 'image', 0, len(images_paths) - 1, nothing)

# Set default value for MAX & MIN HSV trackbars.
cv2.setTrackbarPos('HMin', 'image', 55)
cv2.setTrackbarPos('SMin', 'image', 160)
cv2.setTrackbarPos('VMin', 'image', 10)
cv2.setTrackbarPos('HMax', 'image', 80)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 210)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0
idx = 0

while True:
    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'image')
    sMin = cv2.getTrackbarPos('SMin', 'image')
    vMin = cv2.getTrackbarPos('VMin', 'image')

    hMax = cv2.getTrackbarPos('HMax', 'image')
    sMax = cv2.getTrackbarPos('SMax', 'image')
    vMax = cv2.getTrackbarPos('VMax', 'image')

    idx = cv2.getTrackbarPos('Idx', 'image')
    img = cv2.imread(images_paths[idx])

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)

    # Print if there is a change in HSV value
    if (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin, sMin, vMin, hMax, sMax, vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display output image
    cv2.imshow('image', output)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(waitTime) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()