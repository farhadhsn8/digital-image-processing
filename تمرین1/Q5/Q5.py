import cv2
import matplotlib.pyplot as plt




lena = cv2.imread("../supplement/lena.png",cv2.IMREAD_UNCHANGED)
	
print(lena.shape)  # (512, 512)
# plt.imshow(lena)
# plt.show()

otsu_threshold, image_result = cv2.threshold(lena, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


print("Obtained threshold: ", otsu_threshold)    #  Obtained threshold:  117.0
plt.imshow(image_result,cmap='gray')
plt.show()
cv2.imwrite('otsu_threshold117.png', image_result)


