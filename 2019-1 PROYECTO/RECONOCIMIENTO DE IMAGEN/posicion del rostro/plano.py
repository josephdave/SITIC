import cv2
import random
img = cv2.imread('plano.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("thresh", thresh)

mor_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (3, 3), iterations=3)

contours, hierarchy = cv2.findContours(mor_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # I addapted this part of the code. This is how my version works (2.4.16), but it could be different for OpenCV 3 

sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

for c in sorted_contours[1:]:
    area = cv2.contourArea(c)
    if area > 6000:
        print (area)
        cv2.drawContours(img, [c], -1, (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 3)
        x, y, w, h = cv2.boundingRect(c) # the lines below are for getting the approximate center of the rooms
        cx = x + w / 2
        cy = y + h / 2
        cv2.putText(img,str(area),(cx,cy), cv2.FONT_HERSHEY_SIMPLEX, .5,(255,0,0),1,cv2.CV_AA)

cv2.imshow("mor_img", mor_img)
cv2.imshow("img", img)
cv2.waitKey(0)
