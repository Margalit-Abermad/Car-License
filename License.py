import cv2
import imutils
import numpy as np
import pytesseract
# from PIL import Image, ImageEnhance, ImageFilter

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread(r'C:\Users\user\Music\PLicense\images\car4.jpg')
#, cv2.IMREAD_COLOR)
img = cv2.resize(img, (600, 400))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale
#עד כאן עושה אפור
cv2.imshow("gray", gray)
cv2.waitKey(0)
gray = cv2.bilateralFilter(gray, 13, 15, 15)
#עד כאן מטשטש חלקים לא רצויים
cv2.imshow("gray", gray)
cv2.waitKey(0)
edged = cv2.Canny(gray, 30, 200) #Perform Edge Detection
cv2.imshow("gray", gray)
cv2.waitKey(0)
#חיפוש קווי מיתאר בתמונה
contours=cv2.findContours(edged.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
screenCnt = None
cv2.imshow("edged",edged);
cv2.waitKey(0)

#לקיחת 4 קצוות
for c in contours:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break


#מורידה את החלק שלא לוחית רישוי
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)
cv2.imshow("new_image",new_image);
cv2.waitKey(0)


#חותכת את הלוחית והופכת לשחור לבן
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]
cv2.imshow("Cropped",Cropped);
cv2.waitKey(0)

#קריאת מס' רכב
#text = pytesseract.image_to_string(Cropped, config='--psm 11')
#text = pytesseract.image_to_string(Cropped, lang='eng')
text=pytesseract.image_to_string(Cropped, lang='eng', config='--psm 6') #6,7,8,9,10,13
print(" License= ",text)
