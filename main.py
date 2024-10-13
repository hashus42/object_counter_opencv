import cv2
import numpy as np

img1 = cv2.imread("sources/img1.png")
img2 = cv2.imread("sources/img2.png")

WIN_W = 600
WIN_H = 450
SCREEN_W = 1920
SCREEN_H = 1080

img1, img2 = cv2.resize(img1, (WIN_W, WIN_H)), cv2.resize(img2, (600, 450))

min_contour_width = 40
min_contour_height = 40
offset = 10
line_height = 550
matches = []
objects = 0
x = 0
y = 0
w = 0
h = 0
padding = 5


def get_centrolid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1
    return cx, cy


d = cv2.absdiff(img1, img2)
grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(grey, (5, 5), 0)

ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
dilated = cv2.dilate(th, np.ones((3, 3)))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
contours, h = cv2.findContours(
    closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


while True:
    # go through all contours and draw a rectangle around them
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        contour_valid = (w >= min_contour_width) and (
                h >= min_contour_height)

        if not contour_valid:
            continue
        centrolid = get_centrolid(x, y, w, h)
        matches.append(centrolid)
        cv2.circle(img2, centrolid, 5, (0, 255, 0), -1)
        cx, cy = get_centrolid(x, y, w, h)
        for (x, y) in matches:
            if (line_height + offset) > y > (line_height - offset):
                objects = objects + 1
                matches.remove((x, y))

    cv2.rectangle(img2, (x - padding, y - padding), (x + w + padding, y + h + padding), (255, 0, 0), 2)

    # location = ((SCREEN_W // 2) - (WIN_W // 2), 0)
    location = (1250, 225)
    cv2.namedWindow("mainFrame")
    cv2.moveWindow("mainFrame", location[0], location[1])
    cv2.imshow("mainFrame", img2)

    grey_rgb = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)
    cv2.putText(grey_rgb, "GREY", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1)
    th_rgb = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
    cv2.putText(th_rgb, "THRESHOLD", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1)
    dilated_rgb = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
    cv2.putText(dilated_rgb, "DILATED", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1)
    closing_rgb = cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB)
    cv2.putText(closing_rgb, "CLOSING", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1)
    combinedImgR1 = np.concatenate((grey_rgb, th_rgb), axis=1)
    combinedImgR2 = np.concatenate((dilated_rgb, closing_rgb), axis=1)
    combinedImg = np.concatenate((combinedImgR1, combinedImgR2), axis=0)

    # location2 = ((SCREEN_W // 2) - (WIN_W // 2) - 600, SCREEN_H - WIN_H)
    location2 = (0, 0)
    cv2.namedWindow("filters")
    cv2.moveWindow("filters", location2[0], location2[1])
    cv2.imshow("filters", combinedImg)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()
