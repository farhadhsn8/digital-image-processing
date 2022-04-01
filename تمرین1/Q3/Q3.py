import os
import cv2
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np



sample1 = cv2.imread("../supplement/sample_1.jpg",cv2.IMREAD_UNCHANGED)
sample2 = cv2.imread("../supplement/sample_2.jpg",cv2.IMREAD_UNCHANGED)
sample1 = cv2.cvtColor(sample1, cv2.COLOR_BGR2RGB)
sample2 = cv2.cvtColor(sample2, cv2.COLOR_BGR2RGB)
# plt.imshow(sample2,cmap='gray')
# plt.show()


# axes[0].set_title(bulbasaur.name)



def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
    return cv2.LUT(image, table)



def make_gamma_img(img,name):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for gamma in np.arange(0.0, 3.5, 0.5):
        gamma = gamma if gamma > 0 else 0.1
        adjusted = adjust_gamma(img, gamma=gamma)
        cv2.putText(adjusted, "g={}".format(gamma), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        im =  np.hstack([img, adjusted])
        cv2.imshow("Images",im)
        cv2.waitKey(0)
        cv2.imwrite(name+'_gamma-'+str(gamma)+'.jpg', im)


make_gamma_img(sample1,'sample1')
make_gamma_img(sample2,'sample2')











sample1_adjusted = adjust_gamma(sample1, gamma=1.5)
sample2_adjusted = adjust_gamma(sample2, gamma=2.0)


print(sample1.shape ,sample2.shape )   # (683, 512, 3) (683, 512, 3)

fig, subplt = plt.subplots(4,4,figsize=(16,5))

for i in range(0,4):
    if i==0 :
        img = sample1
    if i==1 :
        img = sample2
    if i==2 :
        img = sample1_adjusted
    if i==3 :
        img = sample2_adjusted

    subplt[i][0].imshow(img)
    subplt[i][0].set_title("org image")

    plt.subplot(4,4,(i+1)*4 - 2)
    hist1 = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.bar(range(256),hist1[:,0])
    subplt[i][1].set_title("R ")

    plt.subplot(4,4,(i+1)*4 - 1 )
    hist1 = cv2.calcHist([img],[1],None,[256],[0,256])
    plt.bar(range(256),hist1[:,0])
    subplt[i][2].set_title("G ")

    plt.subplot(4,4,(i+1)*4)
    hist1 = cv2.calcHist([img],[2],None,[256],[0,256])
    plt.bar(range(256),hist1[:,0])
    subplt[i][3].set_title("B ")




plt.show()
fig.savefig('Q3__full_figure.png')
