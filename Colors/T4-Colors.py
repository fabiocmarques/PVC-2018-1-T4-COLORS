import cv2
import numpy as np
import imutils
import argparse
import glob
from scipy.stats import itemfreq


def vectorized_form(img):
    B, G, R = [img[:, :, x] for x in range(3)]
    delta15 = np.abs(R.astype(np.int8) - G.astype(
        np.int8)) > 15  # watch out for np.abs(R-G): because of the UNsigned numbers, they could get clipped!
    more_R_than_B = (R > B)
    is_skin_coloured_during_daytime = ((R > 95) & (G > 40) & (B > 20) &
                                       (img.ptp(axis=-1) > 15) & delta15 & (R > G) & more_R_than_B & (
                                               (R <= 250) & (G <= 250) & (B <= 250)) & (
                                               (R >= 5) & (G >= 5) & (B >= 5)))
    is_skin_coloured_under_flashlight = ((R > 220) & (G > 210) & (B > 170) &
                                         ~delta15 & more_R_than_B & (G > B) & ((R <= 250) & (G <= 250) & (B <= 250)) & (
                                                 (R >= 5) & (G >= 5) & (B >= 5)))
    return np.logical_or(is_skin_coloured_during_daytime, is_skin_coloured_under_flashlight)


def addValue(p):
    if p > 0:
        # print(p)
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

    return [int(Bfiltered.mean()), int(Gfiltered.mean()), int(Rfiltered.mean())]


funcPixelNonZero = np.vectorize(addValue)


def main1():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--full", help="Use the flags to set the execution type.", action="store_true")
    args = ap.parse_args()

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

    for imageName in glob.glob(images + "/Train/*.jpg"):
        frame = cv2.imread(imageName)
        mapF = vectorized_form(frame)
        mapBG = np.uint8(np.where(mapF, 255, 0))
        newImage = cv2.bitwise_and(frame, frame, mask=mapBG)
        # print(mapBG)
        cv2.imshow("images", newImage)
        key = cv2.waitKey(0)
        # if the 'q' key is pressed, stop the loop
        if key & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    return


def main2():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--full", help="Use the flags to set the execution type.", action="store_true")
    args = ap.parse_args()

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

    # define the upper and lower boundaries of the HSV pixel
    # intensities to be considered 'skin'
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    for imageName in glob.glob(images + "/Test/*.jpg"):
        # grab the current frame
        frame = cv2.imread(imageName)

        # resize the frame, convert it to the HSV color space,
        # and determine the HSV pixel intensities that fall into
        # the speicifed upper and lower boundaries
        frame = imutils.resize(frame, width=400)
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)

        # apply a series of erosions and dilations to the mask
        # using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)

        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skin = cv2.bitwise_and(frame, frame, mask=skinMask)

        # show the skin in the image along with the mask
        cv2.imshow("images", np.hstack([frame, skin]))

        # if the 'q' key is pressed, stop the loop
        key = cv2.waitKey(0)
        if key & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

main1()
#main2()
