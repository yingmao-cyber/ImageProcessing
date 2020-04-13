import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# from PIL import Image

'''
## --- Read and Save Image
# img = cv2.imread('Strining.JPG',cv2.IMREAD_GRAYSCALE)
img = cv2.imread('Strining.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img)
print(np.size(img))
print(gray)
cv2.imshow('image',img)
cv2.imshow('imagegrey',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('watch.png', img)
# plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.plot([50, 100], [80, 100], 'c', linewidth=5)
# plt.show()
'''
'''
## --- Capture Video
cap = cv2.VideoCapture(0)
# codec: compressor and decompressor used to compress and decompress the video information from original source
# Different codecs have different ways to store color, space info, but it also reaults in various quality and speed
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
i = 0

while True:
    # ret: return will be True or False
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # out.write(frame)
    cv2.imshow('frame', frame)
    # cv2.imshow('gray', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('watch{}.png'.format(i+1), frame)
        i = i + 1
cap.release()
# out.release()
cv2.destroyAllWindows()
'''

'''
## --- Draw and Write on Images Using OpenCV Functions
img = cv2.imread('Strining.JPG',cv2.IMREAD_COLOR)
# cv2.line(source, start_of_line, end_of_line, color_range, width of line)
cv2.line(img, (0,0), (150,150), (255, 0, 0), 15)
# cv2.rectangle(source, start_of_top_left_corner, end_of_bottom_right_corner, color_range, width of line)
cv2.rectangle(img, (15,25), (200,150), (255,255,255), 5)
# cv2.circle(source, center_of_circle, radius_of_circle, color_range, width of line)
cv2.circle(img, (200,63), 55, (0,255,0), -1)
# cv2.polylines(source, array_of_points, True(if link the start point and end point) or False, color, width)
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10], [40, 60]], np.int32)
cv2.polylines(img, [pts], False, (0, 255, 255), 3)
# cv2.putText(img, text, start_position, font, size, color, thickness, cv2.LINE_AA)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV Tuts', (300, 130), font, 1, (200, 250, 250), 2, cv2.LINE_AA)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
## --- Image Analysis
img = cv2.imread('Strining.JPG', cv2.IMREAD_COLOR)
# px = img[55, 55] = img[x_coord, y_coord]
# for i in xrange(100):
#     img[55+i, 55] = [0, 255, 0]
# roi = img[300:450, 300:450]
# img[300:450, 300:450] = [0, 255, 0]
img[300:450, 300:450] = img[400:550, 400:550]
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
## --- Image Arithmetic and Logical Operations
img1 = cv2.imread('castle_image.jpg')
img2 = cv2.imread('dog_image.jpg')
img3 = cv2.imread('logo_image.jpg')
## img1 + img2 each image remain its own color
# add = img1[0:280, 0:378] + img2
## cv2.add added pixel value
# add = cv2.add(img1[0:280, 0:378], img2)
# weighted = cv2.addWeighted(img1[0:280, 0:378], 0.7, img2, 0.3, 0)
rows, cols, channels = img3.shape
roi_img1 = img1[0:rows, 0:cols]
# img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
## If the color pixel above 220, it will be converted to 255
# ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)
# mask_inv = cv2.bitwise_not(mask)
# img1_bg = cv2.bitwise_and(roi_img1, roi_img1, mask=mask_inv)
# img2_fg = cv2.bitwise_or(img2, img2, mask=mask_inv)
# dst = cv2.add(img1_bg, img2_fg)
# img1[0:rows, 0:cols] = dst
img3gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img3gray, 80, 255, cv2.THRESH_BINARY_INV)
mask_inv = cv2.bitwise_not(mask)
img1_bg = cv2.bitwise_and(roi_img1, roi_img1, mask=mask)
img3_fg = cv2.bitwise_and(img3, img3, mask=mask_inv)
add = cv2.add(img1_bg, img3_fg)
add_rows, add_cols, channels = add.shape
add = add[30:add_rows-30, 35:add_cols-35]
img1[30:add_rows-30, 35:add_cols-35] = add
cv2.imshow('img1_bg', img1_bg)
cv2.imshow('img3_fg', img3_fg)
cv2.imshow('image', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
## Thresholding
img = cv2.imread('dog_image.jpg')
retval, threshold = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retva2, threshold2 = cv2.threshold(grayscaled, 150, 255, cv2.THRESH_BINARY)
# adaptiveThreshold(src, dst, maxValue, adaptiveMethod, thresholdType, blockSize, C)
gaus = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 2)
cv2.imshow('original', img)
cv2.imshow('image', threshold)
cv2.imshow('gaus', gaus)
cv2.imshow('threshold2', threshold2)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
## Filtering
cap = cv2.VideoCapture(0)
ilowH = 0
ihighH = 179

ilowS = 0
ihighS = 255
ilowV = 0
ihighV = 255

cv2.namedWindow('image')
def callback(x):
    pass
# create trackbars for color change
cv2.createTrackbar('lowH', 'image', ilowH, 179, callback)
cv2.createTrackbar('highH', 'image', ihighH, 179, callback)

cv2.createTrackbar('lowS', 'image', ilowS, 255, callback)
cv2.createTrackbar('highS', 'image', ihighS, 255, callback)

cv2.createTrackbar('lowV', 'image', ilowV, 255, callback)
cv2.createTrackbar('highV', 'image', ihighV, 255, callback)


while True:
    # ret: return will be True or False
    _, frame = cap.read()
    # hue, value, saturation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ilowH = cv2.getTrackbarPos('lowH', 'image')
    ihighH = cv2.getTrackbarPos('highH', 'image')
    ilowS = cv2.getTrackbarPos('lowS', 'image')
    ihighS = cv2.getTrackbarPos('highS', 'image')
    ilowV = cv2.getTrackbarPos('lowV', 'image')
    ihighV = cv2.getTrackbarPos('highV', 'image')
    # hsv hue sat value
    lower_hsv = np.array([ilowH, ilowS, ilowV])
    higher_hsv = np.array([ihighH, ihighS, ihighV])
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
    # if it is in the range, mask will be 1
    # so if it is within the frame, the true color will show, otherwise black
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
'''

'''
## --- Blur and smoothed
kernal = np.ones((15, 15), np.float32)/255
smoothed = cv2.filter2D(res, -1, kernal)
blur = cv2.GaussianBlur(res, (15, 15), 0)
median = cv2.medianBlur(res, 15)
bilateral = cv2.bilateralFilter(res, 15, 75, 75)
'''

'''
## --- Erosion, Dilation, Opening, Closing
kernel = np.ones((5, 5), np.uint8)
# to make pixels within the size to be identical; if not, it will get rid of the pixel that is not
erosion = cv2.erode(mask, kernel, iterations=1)
# instead of making particular pixels to be identical to others, it changes other pixel within the range
dilation = cv2.dilate(mask, kernel, iterations=1)
# opening: remove stuff from the background (remove false positivies)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# closing: remove noise in the object, remove false positives
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
'''


## --- edge detection
cap = cv2.VideoCapture(0)
while True:
    # ret: return will be True or False
    _, frame = cap.read()
    grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(grayscaled, cv2.CV_64F)
    # src, data_type, x, y, size
    # sobelx = cv2.Sobel(grayscaled, cv2.CV_64F, 1, 0, kszie=5)
    # sobely = cv2.Sobel(grayscaled, cv2.CV_64F, 0, 1, kszie=5)
    # edge detection
    edges = cv2.Canny(frame, 50, 50)
    cv2.imshow('laplacian', laplacian)
    cv2.imshow('sobely', edges)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

# hough line -> edge detection & particle detection

'''
## --- Template Matching
img_bgr = cv2.imread('strining.jpg')
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
template = cv2.imread('stringing_bar.JPG', 0)
w, h  = template.shape[::-1]
# print(w, h)
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.9
loc = np.where(res >= threshold)
# print(loc)
for pt in zip(*loc[::-1]):
    # cv2.circle(img_bgr, pt, 1, (0, 255, 255), -1)
    cv2.rectangle(img_bgr, pt, (pt[0]+w, pt[1]+h), (0, 255, 255), 2)
cv2.imshow('detected',img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
## -- Foreground extraction
img = cv2.imread('castle_image.jpg')
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (50, 50, 200, 250)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
plt.imshow(img)
plt.colorbar()
plt.show()
'''

'''
## -- Corner Detection
img = cv2.imread('Strining.jpg')
print(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 1)
corners = np.int0(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 3, 255, -1)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
## Feature Matching
# orb.detectAndCompute
# BFMathcer
# sorted to sort the match
# cv2.drawMatches
'''

'''
## --- Motion Detection
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
while True:
    # ret: return will be True or False
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', fgmask)
    cv2.imshow('original', frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
'''

'''
## -- Haar cascade
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
# eye_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_smile.xml')
# eye_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # frame has to be gray for cascade to work
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    # faces = eye_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        # cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
        # cv2.line(frame, (x, y), (x+w, y+h), (255, 0, 0), 15)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        cv2.imwrite('ying_face2.png', roi_color)

        # eyes = eye_cascade.detectMultScale(roi_gray)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=50)
        for (ex, ey, ew, eh) in smile:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

'''
'''
## --- Face Recognizer - face train
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "face_recog_train")
face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face_LBPHFaceRecognizer.create()
current_id = 0
label_ids = {}
y_lables = []
x_train = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ","-").lower()
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # y_lables.append(label)
            # x_train.append(path)
            ## convert an image into numbers
            pil_image = Image.open(path).convert("L") # convert to gray scale
            image_array = np.array(pil_image, "uint8")
            faces = face_cascade.detectMultiScale(image_array)
            for (x, y, w, h) in faces:
                roi = image_array[y:y + h, x:x + w]
                x_train.append(roi)
                y_lables.append(id_)
recognizer.train(x_train, np.array(y_lables))
recognizer.save("trainer.yml")
# print(x_train)
# print(label_ids)


## -- Face Recognizer - face find
face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face_LBPHFaceRecognizer.create()
recognizer.read("trainer.yml")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # frame has to be gray for cascade to work
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    labels = {'ying': 3, 'qiuzhang': 2, 'harryporter': 0, 'hermione': 1}
    labels = {v:k for k,v in labels.items()}


    for (x, y, w, h) in faces:
        # cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
        # cv2.line(frame, (x, y), (x+w, y+h), (255, 0, 0), 15)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45:
            print(labels[id_])

    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
'''