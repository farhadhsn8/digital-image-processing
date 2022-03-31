import os
import cv2
import matplotlib.pyplot as plt

def generate_bitplane_slicing(image):

    B0 = image % 2.0
    t= image - B0

    B1 =  (t % 4 ) / 2 
    t= t - 2 * B1

    B2 = (t % 8 ) / 4
    t= t - 4 * B2

    B3 = (t % 16 ) / 8 
    t= t - 8 * B3

    B4 = (t % 32 ) / 16
    t= t - 16 * B4

    B5 = (t % 64 ) / 32 
    t= t - 32 * B5

    B6 = (t % 128 ) / 64 
    t= t - 64 * B6

    B7 = (t % 256 ) / 128 
    
    return B0 , B1 , B2 , B3 , B4 , B5 , B6 , B7 




baboon = cv2.imread("../supplement/baboon.png",cv2.IMREAD_UNCHANGED)
cameraman = cv2.imread("../supplement/cameraman.tif",cv2.IMREAD_UNCHANGED)
baboon = cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB)
cameraman = cv2.cvtColor(cameraman, cv2.COLOR_BGR2RGB)
# plt.imshow(baboon,cmap='gray')
# plt.show()

baboon_bitPlane = generate_bitplane_slicing(baboon)
cameraman_bitPlane = generate_bitplane_slicing(cameraman)

steganoCameraman = cameraman - cameraman_bitPlane[0] + baboon_bitPlane[7]

steganoCameraman_bitPlane = generate_bitplane_slicing(steganoCameraman)

fig, subplt = plt.subplots(3,9,figsize=(16,5))
subplt[0][0].imshow(baboon)
subplt[0][0].set_title("org image")
subplt[0][1].imshow(baboon_bitPlane[0])
subplt[0][1].set_title("bit 0")
subplt[0][2].imshow(baboon_bitPlane[1])
subplt[0][2].set_title("bit 1")
subplt[0][3].imshow(baboon_bitPlane[2])
subplt[0][3].set_title("bit 2")
subplt[0][4].imshow(baboon_bitPlane[3])
subplt[0][4].set_title("bit 3")
subplt[0][5].imshow(baboon_bitPlane[4])
subplt[0][5].set_title("bit 4")
subplt[0][6].imshow(baboon_bitPlane[5])
subplt[0][6].set_title("bit 5")
subplt[0][7].imshow(baboon_bitPlane[6])
subplt[0][7].set_title("bit 6")
subplt[0][8].imshow(baboon_bitPlane[7])
subplt[0][8].set_title("bit 7")
subplt[1][0].imshow(cameraman)
subplt[1][1].imshow(cameraman_bitPlane[0])
subplt[1][2].imshow(cameraman_bitPlane[1])
subplt[1][3].imshow(cameraman_bitPlane[2])
subplt[1][4].imshow(cameraman_bitPlane[3])
subplt[1][5].imshow(cameraman_bitPlane[4])
subplt[1][6].imshow(cameraman_bitPlane[5])
subplt[1][7].imshow(cameraman_bitPlane[6])
subplt[1][8].imshow(cameraman_bitPlane[7])
subplt[2][0].imshow(steganoCameraman.astype(int))
subplt[2][1].imshow(steganoCameraman_bitPlane[0])
subplt[2][2].imshow(steganoCameraman_bitPlane[1])
subplt[2][3].imshow(steganoCameraman_bitPlane[2])
subplt[2][4].imshow(steganoCameraman_bitPlane[3])
subplt[2][5].imshow(steganoCameraman_bitPlane[4])
subplt[2][6].imshow(steganoCameraman_bitPlane[5])
subplt[2][7].imshow(steganoCameraman_bitPlane[6])
subplt[2][8].imshow(steganoCameraman_bitPlane[7])
plt.show()
fig.savefig('full_figure.png')