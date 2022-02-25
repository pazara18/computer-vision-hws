import numpy as np
import cv2
import pyautogui
import time

time.sleep(5)
end = time.time() + 60
while end > time.time():
    screenshot = pyautogui.screenshot()

    image = np.array(screenshot)
    image = image[:, :, ::-1].copy()
    image = image[int(0.75 * image.shape[0]):image.shape[0], int(0.40 * image.shape[1]):int(0.60 * image.shape[1])]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 1:
        approx = cv2.approxPolyDP(contours[1], 0.01 * cv2.arcLength(contours[1], True), True)
        if len(approx) == 3:
            pyautogui.press('a')
        elif len(approx) == 4:
            pyautogui.press('s')
        elif len(approx) == 6:
            pyautogui.press('f')
        elif len(approx) == 10:
            pyautogui.press('d')
