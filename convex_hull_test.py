import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


im = Image.open("FAKECELLS.jpg").convert("1")
im.show()
print(im.size)
arr = np.array(im)
fauxrray = []
for a in range(0, (im.size[0])):
    for b in range(0, im.size[1]):
        if arr[b, a] == True:
            fauxrray.append([[a, b]])

# print(len(fauxrray))
# print(fauxrray)

img = cv2.imread("FAKECELLS.jpg")
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,10,255,0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# img2 = cv2.drawContours(img, np.asarray(fauxrray), -1, (1,255,0), 3)
# plt.imshow(img2, cmap = 'gray', interpolation = 'bicubic')
# plt.show()

hull = cv2.convexHull(np.asarray(fauxrray))
print(hull)

img4 = cv2.drawContours(img, hull, -1, (0,255,0), 3)
img6 = cv2.fillConvexPoly(img, hull, color = (0,255,0))
plt.imshow(img6, cmap = 'gray', interpolation = 'bicubic')
plt.show()


### seems like it doesn't recognise some parts as a valid part of the hull (won't go into the inny-parts)





### this code tries to collect the array another way...
# img = cv2.imread("FAKECELLS.jpg")
# imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imgray,10,255,0)
# image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
# # img2 = cv2.drawContours(img, contours, -1, (1,255,0), 3)
# # plt.imshow(img2, cmap = 'gray', interpolation = 'bicubic')
# # plt.show()
#
# combotours = []
# for x in range(len(contours)):
#     a = contours[x]
#     for item in a:
#         combotours.append(item)
#
# # print(np.asarray(combotours))
#
# # img3 = cv2.drawContours(img, np.asarray(combotours), -1, (0,255,0), 3)
# # plt.imshow(img3, cmap = 'gray', interpolation = 'bicubic')
# # plt.show()
#
# hull = cv2.convexHull(np.asarray(combotours))
# print(hull)
# img4 = cv2.drawContours(img, hull, -1, (0,255,0), 3)
# # img5 = cv2.polylines(img, hull, isClosed = True, color = (0,255,0), ncontours = 5, thickness = 10)
# img6 = cv2.fillConvexPoly(img, hull, color = (0,255,0))
#
#
# plt.imshow(img6, cmap = 'gray', interpolation = 'bicubic')
# plt.show()