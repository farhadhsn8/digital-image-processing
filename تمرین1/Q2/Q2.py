import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


peppers = cv2.imread("../supplement/peppers.png",cv2.IMREAD_UNCHANGED)
peppers = cv2.cvtColor(peppers, cv2.COLOR_BGR2RGB)


peppers_hsv = cv2.cvtColor(peppers, cv2.COLOR_RGB2HSV)
peppers_ycbcr = cv2.cvtColor(peppers, cv2.COLOR_RGB2YCR_CB)

print(peppers_ycbcr.shape)


fig, subplt = plt.subplots(3,4,figsize=(16,5))
subplt[0][0].imshow(peppers)
subplt[0][0].set_title("image RGB")
subplt[0][1].imshow(peppers[:,:,0],cmap='gray')
subplt[0][1].set_title("R")
subplt[0][2].imshow(peppers[:,:,1],cmap='gray')
subplt[0][2].set_title("G")
subplt[0][3].imshow(peppers[:,:,2],cmap='gray')
subplt[0][3].set_title("B")

subplt[1][0].imshow(peppers_hsv)
subplt[1][0].set_title("image HSV")
subplt[1][1].imshow(peppers_hsv[:,:,0],cmap='gray')
subplt[1][1].set_title("H")
subplt[1][2].imshow(peppers_hsv[:,:,1],cmap='gray')
subplt[1][2].set_title("S")
subplt[1][3].imshow(peppers_hsv[:,:,2],cmap='gray')
subplt[1][3].set_title("V")

subplt[2][0].imshow(peppers_ycbcr)
subplt[2][0].set_title("image YCbCr")
subplt[2][1].imshow(peppers_ycbcr[:,:,0],cmap='gray')
subplt[2][1].set_title("Y")
subplt[2][2].imshow(peppers_ycbcr[:,:,1],cmap='gray')
subplt[2][2].set_title("Cb")
subplt[2][3].imshow(peppers_ycbcr[:,:,2],cmap='gray')
subplt[2][3].set_title("Cr")

plt.show()
fig.savefig('Q2_full_figure.png')


def rgb2ycbcr(im):
    cbcr = np.empty_like(im)
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]
    # Y
    cbcr[:,:,0] = .299 * r + .587 * g + .114 * b
    # Cb
    cbcr[:,:,1] = 128 - .169 * r - .331 * g + .5 * b
    # Cr
    cbcr[:,:,2] = 128 + .5 * r - .419 * g - .081 * b
	
    return np.uint8(cbcr)



def rgb_to_hsv(r, g, b):

	r, g, b = r / 255.0, g / 255.0, b / 255.0

	cmax = max(r, g, b) 
	cmin = min(r, g, b) 
	diff = cmax-cmin	

	if cmax == cmin:
		h = 0
	
	elif cmax == r:
		h = (60 * ((g - b) / diff) + 360) % 360

	elif cmax == g:
		h = (60 * ((b - r) / diff) + 120) % 360

	elif cmax == b:
		h = (60 * ((r - g) / diff) + 240) % 360

	if cmax == 0:
		s = 0
	else:
		s = (diff / cmax) * 100

	v = cmax * 100
	return h, s, v






a = np.array([[[87,200,17]]])

print('ycbcr: ',rgb2ycbcr(a))    #  [[[145  55  86]]]
print('hsv : ',rgb_to_hsv(87,200,17))   #  (97.04918032786885, 91.5, 78.43137254901961)