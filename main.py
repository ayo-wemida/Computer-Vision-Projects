# This python script is able to identify label and count the number of
# defects in the gears in the attached image.

import cv2 as cv
import os
import numpy as np


Gears = cv.imread("Color Gears.jpg")
#print(Gears.shape)

gs_Gears = cv.cvtColor(Gears, cv.COLOR_BGR2GRAY)

#print(Gears[113, 100:300, :]) ###Used this to visualize the range of values for the red channel

#ret, bin_gears = cv.threshold(gs_Gears, 190, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

bin_gears = np.zeros(gs_Gears.shape, dtype= np.uint8)

####Use "list slicing" to access the pixels in the color image where the red channel is less than 60

bin_gears[Gears[:, :, 2] < 60] = 255
###Now cover the holes

# Copy the thresholded image to a floodfill variable
im_floodfill = bin_gears.copy()
# Notice the size needs to be 2 pixels than the image
h, w = bin_gears.shape[:2]
print( h, w)
mask = np.zeros((h + 2, w + 2), np.uint8)
# Floodfill from point (0, 0)
cv.floodFill(im_floodfill, mask, (0, 0), 255)
# Invert floodfilled image
im_floodfill_inv = cv.bitwise_not(im_floodfill)
# Combine the two images to get the foreground.
im_out = bin_gears | im_floodfill_inv

##Now isolate the teeth of the gears

##Try different operations on im out
kernel= cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
img_erosion= cv.erode(im_out,kernel,iterations=2)
Gear_size = 130
####Remove teeth and create a ring of gear body
Gear_Body_Mask = cv.getStructuringElement(cv.MORPH_ELLIPSE, (Gear_size, Gear_size))
Remove_Gear_Teeth = cv.morphologyEx(im_out, cv.MORPH_OPEN, Gear_Body_Mask)
Expand = cv.dilate(Remove_Gear_Teeth,kernel,iterations=8)
##Get the gear body ring
Ring = cv.bitwise_xor(Remove_Gear_Teeth,Expand)
#cv.imshow("ring",Ring)

####Edge here will be just the teeth
Edge = cv.bitwise_xor(im_out,img_erosion)
cv.imshow("edge",Edge)
kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=np.uint8)
###Dilate teeth with diamond kernel
Edge_dilate = cv.dilate(Edge, kernel, iterations=2)


#Try different operations till you get best hole isolation
holes= cv.bitwise_xor(Edge_dilate,Ring)
kernel= cv.getStructuringElement(cv.MORPH_ELLIPSE,(6,6))
holes = cv.morphologyEx(holes, cv.MORPH_OPEN, kernel,iterations=1)
holes = cv.morphologyEx(holes, cv.MORPH_CLOSE,kernel,iterations=2)
kernel= cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
holes = cv.dilate(holes,kernel)

##Use CCL to label the holes
num_labels, Image_Labels = cv.connectedComponents(holes)
max_labels= np.max(Image_Labels)
#Check how many labels you have with the maximum integer label
print(max_labels)


####Perform operations to give each label a color
label_hue = np.uint8(200 * Image_Labels / np.max(Image_Labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_image = cv.merge([label_hue, blank_ch, blank_ch])
labeled_image = cv.cvtColor(labeled_image, cv.COLOR_HSV2BGR)
####convert the background color to black
labeled_image[label_hue == 0] = 0
cv.imshow("labeled_img",labeled_image)

#Convert binary image to BGR so you can add color label to  image
bin_gears= cv.cvtColor(bin_gears,cv.COLOR_GRAY2BGR)
img_final = labeled_image+bin_gears

cv.putText(img_final, "EEGR 675: Project 1", (10, 15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (100, 250, 100), 1, cv.LINE_AA)
cv.putText(img_final, str(max_labels) + " Defects found", (10, 35), cv.FONT_HERSHEY_SIMPLEX, 0.4, (100, 250, 155), 1, cv.LINE_AA)



cv.imshow("Original Gears",Gears)
cv.imshow("Defect Identification",img_final)
cv.waitKey(0)