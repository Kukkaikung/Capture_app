import cv2
import numpy as np

class BrownRectangleDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original = None
        self.image = None
        self.hsv = None
        self.mask = None
        self.contours = []

    def load_image(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        self.original = self.image.copy()
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    def create_mask(self):
        lower_brown = np.array([10, 100, 20])
        upper_brown = np.array([35, 255, 200])
        self.mask = cv2.inRange(self.hsv, lower_brown, upper_brown)

        kernel = np.ones((5, 5), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)

    def find_contours(self):
        self.contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def process_contours(self):
        for cnt in self.contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area > 500:
                    cv2.drawContours(self.image, [approx], -1, (0, 255, 0), 3)

                    pts = approx.reshape(4, 2)
                    rect = self.order_points(pts)
                    self.warp_perspective(rect)

    def warp_perspective(self, rect):
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self.original, M, (maxWidth, maxHeight))
        cv2.imshow("Warped", warped)

    def show_results(self):
        cv2.imshow("Detected Brown Rectangles", self.image)
        cv2.imshow("Mask", self.mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(self):
        self.load_image()
        self.create_mask()
        self.find_contours()
        self.process_contours()
        self.show_results()
