import os
import cv2
import matplotlib.pyplot as plt




cameraman_mnpl = cv2.imread("../supplement/cameraman_mnpl.tif",cv2.IMREAD_UNCHANGED)
cameramanORG = cv2.imread("../supplement/cameraman.tif",cv2.IMREAD_UNCHANGED)
cameraman_mnpl = cv2.cvtColor(cameraman_mnpl, cv2.COLOR_BGR2RGB)
cameramanORG = cv2.cvtColor(cameramanORG, cv2.COLOR_BGR2RGB)

cameraman_mnpl = cv2.medianBlur(cameraman_mnpl, ksize=5)

plt.imshow(cameraman_mnpl)
plt.show()
cv2.imwrite('step1.tif', cameraman_mnpl)

mag_ratio = 0.2
sel_ratio = 0.75
print(int(cameramanORG.shape[1] * (mag_ratio + sel_ratio) ))

for i in range(int(cameramanORG.shape[0] * mag_ratio ),int( cameramanORG.shape[0] * (mag_ratio + sel_ratio))  ):
    for j in range(int(cameramanORG.shape[1] * (mag_ratio + sel_ratio) ), cameraman_mnpl.shape[1] ):
        cameraman_mnpl[int(i - (mag_ratio * cameramanORG.shape[1])), int(j - (mag_ratio * cameramanORG.shape[0]))] = cameraman_mnpl[i,j]

plt.imshow(cameraman_mnpl)
plt.show()
cv2.imwrite('step2.tif', cameraman_mnpl)

for i in range(int(cameramanORG.shape[0] * (mag_ratio + sel_ratio) ), cameraman_mnpl.shape[0] ):
    for j in range(int(cameramanORG.shape[1] * mag_ratio ),int( cameraman_mnpl.shape[1] )  ):
        cameraman_mnpl[int(i - (mag_ratio * cameramanORG.shape[1])), int(j - (mag_ratio * cameramanORG.shape[0]))] = cameraman_mnpl[i,j]


plt.imshow(cameraman_mnpl)
plt.show()
cv2.imwrite('step3.tif', cameraman_mnpl)

cameraman_mnpl = cameraman_mnpl[0:cameramanORG.shape[1] , 0:cameramanORG.shape[0]]

cameraman_mnpl = cv2.blur(cameraman_mnpl,(3,3))

plt.imshow(cameraman_mnpl)
plt.show()
cv2.imwrite('step4.tif', cameraman_mnpl)


psnr = cv2.PSNR(cameraman_mnpl, cameramanORG)
print("psnr is : ", psnr)