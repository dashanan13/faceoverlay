#!/usr/bin/python
# face_replace.py
# Usage: python face_replace.py <image_file> [face index]

import sys
import cv2
from cv2.highgui import *
from PIL import Image, ImageEnhance
import random


def detectObjects(image):
	"""Converts an image to grayscale and returns the locations of any faces found"""
	grayscale = cv2.CreateImage(cvSize(image.width, image.height), 8, 1)
	cv2.CvtColor(image, grayscale, CV_BGR2GRAY)

	storage = cv2.CreateMemStorage(0)
	cv2.ClearMemStorage(storage)
	cv2.EqualizeHist(grayscale, grayscale)
	cascade = cv2.LoadHaarClassifierCascade(
		'/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
		cv2.Size(1,1))

	scalefactor = 1.1  # How much to increase window size each pass
	minsize = 50  # Smallest face to detect. Up this if you have small falsepositives
	faces = cv2.HaarDetectObjects(grayscale, cascade, storage, scalefactor, 50,
				CV_HAAR_DO_CANNY_PRUNING, cvSize(minsize, minsize))

	return [(f.x, f.y, f.x + f.width, f.y + f.height) for f in faces]


def meancol(source):
	"""Find the mean colour of the given image"""
	onepix = source.copy()
	onepix.thumbnail((1,1),Image.ANTIALIAS)
	return onepix.getpixel((0,0))


def adjust(im, col, startcol=None):
	"""Adjust the image such that its mean colour is `col`"""
	if startcol is None:
		startcol = meancol(im)
	rband, gband, bband = im.split()
	rbri, gbri, bbri = ImageEnhance.Brightness(rband), ImageEnhance.Brightness(gband), ImageEnhance.Brightness(bband)
	rband = rbri.enhance((float(col[0]) / float(startcol[0])))
	gband = gbri.enhance((float(col[1]) / float(startcol[1])))
	bband = bbri.enhance((float(col[2]) / float(startcol[2])))
	im = Image.merge("RGB",(rband, gband, bband))
	return im


def main():
	inputfile = sys.argv[1]
	image = cvLoadImage(inputfile)

	print('Detecting faces...')
	faceboxes = detectObjects(image)
	print(len(faceboxes), "faces found")

	im = Image.open(inputfile)
	mask = Image.open("circlemask.png")

	if len(sys.argv) > 2:
		faceindex = int(sys.argv[2])
	else:
		print("Choosing random face")
		faceindex = random.randrange(len(faceboxes))
	print("Using face", faceindex)
	newface = im.crop(faceboxes[faceindex])
	newfacemeancol = meancol(newface)

	for box in faceboxes:
		face = im.crop(box)
		facemeancol = meancol(face)
		adjustedface = adjust(newface, facemeancol, newfacemeancol)

		size = (box[2] - box[0], box[3] - box[1])
		scalednewface = adjustedface.resize(size, Image.ANTIALIAS)
		scaledmask = mask.resize(size, Image.ANTIALIAS)
		im.paste(scalednewface, box, scaledmask)

	im.show()

if __name__ == "__main__":
	main()