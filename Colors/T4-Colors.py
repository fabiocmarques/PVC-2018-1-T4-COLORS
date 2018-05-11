import cv2
import numpy as np
import imutils
import argparse
import glob
from scipy.stats import itemfreq



def addValue(p):
    if p > 0:
        #print(p)
        return int(p)

def RGBMeans(image):
    G = funcPixelNonZero(image[:, :, 1])
    Gflat = G.flatten()
    Gfiltered = Gflat[Gflat != np.array(None)]

    B = funcPixelNonZero(image[:, :, 0])
    Bflat = B.flatten()
    Bfiltered = Bflat[Bflat != np.array(None)]

    R = funcPixelNonZero(image[:, :, 2])
    Rflat = R.flatten()
    Rfiltered = Rflat[Rflat != np.array(None)]



    return [ int(Bfiltered.mean()), int(Gfiltered.mean()), int(Rfiltered.mean()) ]



funcPixelNonZero = np.vectorize(addValue)

def main():
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

    # lowerT = np.array([256, 256, 256], dtype="int32")
    # upperT = np.array([-1, -1, -1], dtype="int32")
    # for imageName in glob.glob(imagesGt + "/Train/*.jpg"):
    #
    #     img = cv2.imread(imageName)
    #     converted = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    #     skinMask = cv2.inRange(converted, 20, 255)
    #     newImage = cv2.bitwise_and(img, img, mask=skinMask)
    #
    #     means = RGBMeans(newImage)
    #
    #     if means[0]+means[1]+means[2] < lowerT[0]+lowerT[1]+lowerT[2]:
    #         lowerT = np.array(means, dtype="int32")
    #     if means[0] + means[1] + means[2] > upperT[0] + upperT[1] + upperT[2]:
    #         upperT = np.array(means, dtype="int32")
    #
    #     average_color = [newImage[:, :, i].mean() for i in range(img.shape[-1])]
    #
    #
    #     arr = np.float32(newImage)
    #     pixels = arr.reshape((-1, 3))
    #
    #     n_colors = 3
    #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    #     flags = cv2.KMEANS_RANDOM_CENTERS
    #     _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    #
    #     palette = np.uint8(centroids)
    #     quantized = palette[labels.flatten()]
    #     quantized = quantized.reshape(img.shape)
    #
    #     #newPalette = palette[palette != np.array([0,0,0], dtype="uint8")]
    #     dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
    #     print(palette)
    #
    #     #cv2.imshow("Image", newImage)
    #     #key = cv2.waitKey(0)
    #     ##   if the 'q' key is pressed, stop the loop
    #     #if key & 0xFF == ord("q"):
    #     #    break
    #
    #
    # cv2.destroyAllWindows()

    lower = cv2.cvtColor(np.uint8([[[30, 30, 30]]]), cv2.COLOR_BGR2HSV).flatten()
    upper = cv2.cvtColor(np.uint8([[[20, 120, 170]]]), cv2.COLOR_BGR2HSV).flatten()


    # lower = np.array([0, 48, 80], dtype = "uint8")
    # upper = np.array([20, 255, 255], dtype = "uint8")
    print("Upper", upper)
    print("Lower", lower)

    for imageName in glob.glob(images+"/Train/*.jpg"):
       frame = cv2.imread(imageName)
       # resize the frame, convert it to the HSV color space,
       # and determine the HSV pixel intensities that fall into
       # the speicifed upper and lower boundaries
       converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
       skinMask = cv2.inRange(converted, lower, upper)
       cv2.imshow("Mask", skinMask)
       cv2.waitKey(0)
       # apply a series of erosions and dilations to the mask
       # using an elliptical kernel
       # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
       # skinMask = cv2.erode(skinMask, kernel, iterations=2)
       # skinMask = cv2.dilate(skinMask, kernel, iterations=2)
       # blur the mask to help remove noise, then apply the
       # mask to the frame
       skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
       skin = cv2.bitwise_and(frame, frame, mask=skinMask)
       # show the skin in the image along with the mask
       cv2.imshow("images", np.hstack([frame, skin]))
       key = cv2.waitKey(0)
       # if the 'q' key is pressed, stop the loop
       if key & 0xFF == ord("q"):
           break
    cv2.destroyAllWindows()
main()