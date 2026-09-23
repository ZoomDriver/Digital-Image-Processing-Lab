import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("test.jpg")
if img is None:
    raise FileNotFoundError("Please check image path!")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(binary, kernel, iterations=1)
dilation = cv2.dilate(binary, kernel, iterations=1)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

contours, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_contour = img.copy()
cv2.drawContours(img_contour, contours, -1, (0, 0, 255), 2)

plt.figure(figsize=(16, 10))

plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(gray, cmap="gray")
plt.title("Gray Image")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(erosion, cmap="gray")
plt.title("Erosion")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(dilation, cmap="gray")
plt.title("Dilation")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(opening, cmap="gray")
plt.title("Opening")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(closing, cmap="gray")
plt.title("Closing")
plt.axis("off")

plt.subplot(2, 4, 8)
plt.imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
plt.title("Contour")
plt.axis("off")

plt.tight_layout()
plt.savefig("result.png", dpi=200, bbox_inches='tight')
plt.show()

print(f"Detected contours: {len(contours)}")
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    print(f"Contour {i+1}, Area: {area}")
