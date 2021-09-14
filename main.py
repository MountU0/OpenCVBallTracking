from collections import deque
import numpy as np
import cv2

cap = cv2.VideoCapture('vids2/240fps.MOV')#your stream or video with blue ball
# I did't test different colors with this settings

pts = deque(maxlen=128)#len of a trace

while cap.isOpened():
    success, img = cap.read()
    cropped = img[350:900, 50:600]

    hsv_img = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 100, 0])
    upper_blue = np.array([255, 255, 255])
    masking = cv2.inRange(hsv_img, lower_blue, upper_blue)
    # canny = cv2.Canny(masking, 30, 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    foreground = cv2.morphologyEx(masking, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.dilate(foreground, kernel, iterations=2)

    ret, thresh = cv2.threshold(dilation, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(img.shape[:-1], np.uint8)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

    center = None

    for contour in contours:
        vect = []
        (x, y, w, h) = cv2.boundingRect(contour)  # converting a mass to tuple
        print("X: ", x + (w / 2), "Y: ", y + (h / 2), " ")

        center = (int(x + (w / 2)), int(y + (h / 2)))
        if cv2.contourArea(contour) < 330:  # max area of a ball
            continue
        cv2.rectangle(cropped, (x, y), (x + w, y + h), (255, 255, 255), 2)  # getting a rectangle from tuple
        vect.clear()

    pts.appendleft(center)

    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and draw the connecting lines
        thickness = int(np.sqrt(128 / float(i + 1)) * 1.5)
        cv2.line(cropped, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.imshow('Tracking', cropped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()