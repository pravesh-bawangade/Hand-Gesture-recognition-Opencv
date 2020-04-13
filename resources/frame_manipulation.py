"""
@Author - Pravesh
This File contains functions to capture frame and return it.
"""
import cv2
import numpy as np


class FrameCapture:

    def __init__(self, source=0, size=(640, 480)):
        self.source = source
        self.size = size
        self.cap = cv2.VideoCapture(self.source)
        self.x_d, self.y_d = size[0], size[1]

    def capture_frame(self, flip=1):
        """
        :param flip: default = 1
        :return: captured Frame
        """
        ret, frame = self.cap.read()
        if not ret:
            return "Not able to capture image from source {0}".format(self.source)
        frame = cv2.flip(frame, flip)
        return  frame

    def resize_frame(self, frame):
        """
        :param shape: shape to which you want to resize the frame
        :return: resized fame
        """
        return cv2.resize(frame, self.size)

    def define_roi(self, frame):
        """
        defines square region of interest
        :param frame: frame from which roi should be defined
        :return: roi : region of interest
                frame : frame with roi defined
                hsv : hsv of roi
        """
        width_start, width_size = int(self.x_d / 6.4), int((self.x_d / 3.2) + (self.x_d / 6.4))
        roi = frame[width_start:width_size, width_start:width_size]

        cv2.rectangle(frame, (width_start, width_start), (width_size, width_size), (0, 255, 0), 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        return frame, roi, hsv

    @staticmethod
    def create_mask(lower_bound, upper_bound, hsv, kernel_size=(3,3)):
        """
        returns desired mask according to the given color range
        :param upper_bound: upper bound color range in hsv
        :param lower_bound: lower bound color range in hsv
        :return: mask : desired mask
        """
        lower_bound = np.array(lower_bound, dtype=np.uint8)
        upper_bound = np.array(upper_bound, dtype=np.uint8)

        kernel = np.ones(kernel_size, np.uint8)
        # extract skin color image
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask, kernel, iterations=4)

        # blur the image
        mask = cv2.GaussianBlur(mask, (5, 5), 100)
        return mask

    @staticmethod
    def get_contours(mask):
        """
        returns contours from mask image
        :param mask: mask from image
        :return: contours
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def show_image(name, image):
        """
        Displays image on screen
        :param image: input image to display
             name: name on window
        :return: None
        """
        cv2.imshow(name, image)

    def close_all(self):
        cv2.destroyAllWindows()
        self.cap.release()