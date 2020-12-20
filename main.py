import cv2
import glob
import numpy as np


def evaluate(img_col):
    np.seterr(all='ignore')
    img_gry = cv2.cvtColor(img_col, cv2.COLOR_RGB2GRAY)
    rows, cols = img_gry.shape
    crow, ccol = rows//2, cols//2
    f = np.fft.fft2(img_gry)
    fshift = np.fft.fftshift(f)
    fshift[crow-75:crow+75, ccol-75:ccol+75] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_fft = np.fft.ifft2(f_ishift)
    img_fft = 20*np.log(np.abs(img_fft))
    result = np.mean(img_fft)
    return img_fft, result, result < 10


cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()

    text = "Not Blurry"
    if evaluate(frame)[2]:
        text = "Blurry"

    cv2.putText(frame, "{}".format(text), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
