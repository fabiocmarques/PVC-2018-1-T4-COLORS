import cv2
import numpy as np
import imutils
import argparse
import glob



def addValue(p):
    #if p[0] > 0 and p[1] > 0 and p[2] > 0:
    if p > 0:
        global pixelValuesR
        global pixelValuesG
        global pixelValuesB

        pixelValuesB = np.concatenate(pixelValuesB, np.array([int(p)], dtype="int32"))
        #np.insert(pixelValuesB, 0, p[0])
        #np.insert(pixelValuesG, 0, p[1])
        #np.insert(pixelValuesR, 0, p[2])

pixelValuesR = np.array([], dtype="int32")
pixelValuesG = np.array([], dtype="int32")
pixelValuesB = np.array([], dtype="int32")
funcPixelNonZero = np.vectorize(addValue)

def main():
    global pixelValuesR
    global pixelValuesG
    global pixelValuesB
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--full", help="Use the flags to set the execution type.", action="store_true")
    args = ap.parse_args()
    
    # define the upper and lower boundaries of the HSV pixel
    # intensities to be considered 'skin'
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    imageNumbers = np.array([], dtype="uint32")
    imagePath = ""
    
    if not args.full:
        imageTrain = np.array([11, 24, 44, 83, 331, 429, 789, 841], dtype="uint32")
        imageResults = np.array([243, 278], dtype="uint32")
        images = "./Images/SkinDataset/ORI"
        imagesGt = "./Images/SkinDataset/GT"
    else:
        imageTrain = np.array(np.arange(1, 783, dtype="uint32"))
        imageConsolidate = np.array(np.arange(783, 951, dtype="uint32"))
        imageResults = np.array(np.arange(951, 1119, dtype="uint32"))
        images = "./Images/ORI"
        imagesGt = "./Images/GT"
    
    lowerMask = np.array([20], dtype="uint8")
    upperMask = np.array([255], dtype="uint8")
    for imageName in glob.glob(imagesGt + "/Train/*.jpg"):
        pixelValuesR = np.array([], dtype="int32")
        pixelValuesG = np.array([], dtype="int32")
        pixelValuesB = np.array([], dtype="int32")

        img = cv2.imread(imageName)
        converted = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        skinMask = cv2.inRange(converted, 10, 255)
        newImage = cv2.bitwise_and(img, img, mask=skinMask)

        funcPixelNonZero(newImage[:, :, 0])

        print("Count", pixelValuesB.size)
        print([pixelValuesB.mean()])
    
        #cv2.imshow("Image", newImage)
        #key = cv2.waitKey(0)
        ##   if the 'q' key is pressed, stop the loop
        #if key & 0xFF == ord("q"):
        #    break
    
    cv2.destroyAllWindows()
    
    #lower = np.array([0, 48, 80], dtype = "uint8")
    #upper = np.array([20, 255, 255], dtype = "uint8")
    #
    #for imageName in glob.glob(images+"/Train/*.jpg"):
    #    frame = cv2.imread(imageName)
    #    # resize the frame, convert it to the HSV color space,
    #    # and determine the HSV pixel intensities that fall into
    #    # the speicifed upper and lower boundaries
    #    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #    skinMask = cv2.inRange(converted, lower, upper)
    #    # apply a series of erosions and dilations to the mask
    #    # using an elliptical kernel
    #    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    #    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    #    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    #    # blur the mask to help remove noise, then apply the
    #    # mask to the frame
    #    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    #    skin = cv2.bitwise_and(frame, frame, mask=skinMask)
    #    # show the skin in the image along with the mask
    #    cv2.imshow("images", np.hstack([frame, skin]))
    #    key = cv2.waitKey(0)
    #    # if the 'q' key is pressed, stop the loop
    #    if key & 0xFF == ord("q"):
    #        break
    #cv2.destroyAllWindows()
main()