from resources import frame_manipulation as frm
import cv2
import math


class HandGesture:

    def __init__(self, source=0,size=(640,420)):
        self.fm = frm.FrameCapture(source,size=size)
        self.contours = None
        self.roi = None
        self.frame = None
        self.mask = None
        self.hsv = None

    def detect_hand(self, lower_skin=[0, 20, 70], upper_skin=[20, 255, 255]):
        self.frame = self.fm.capture_frame()
        self.frame = self.fm.resize_frame(self.frame)
        self.frame, self.roi, self.hsv = self.fm.define_roi(self.frame)
        self.mask = self.fm.create_mask(lower_bound=lower_skin, upper_bound=upper_skin, hsv=self.hsv)
        self.contours = self.fm.get_contours(self.mask)

    def recognize(self):
        try:
            # find contour of max area(hand)
            cnt = max(self.contours, key=lambda x: cv2.contourArea(x))

            # approx the contour a little
            epsilon = 0.0005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # make convex hull around hand
            hull = cv2.convexHull(cnt)

            # define area of hull and area of hand
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(cnt)

            # find the percentage of area not covered by hand in convex hull
            arearatio = ((areahull - areacnt) / areacnt) * 100

            # find the defects in convex hull with respect to hand
            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx, hull)

            # l = no. of defects
            l = 0

            # code for finding no. of defects due to fingers
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                pt = (100, 180)

                # find length of all sides of triangle
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                s = (a + b + c) / 2
                ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

                # distance between point and convex hull
                d = (2 * ar) / a

                # apply cosine rule here
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                if angle <= 90 and d > 30:
                    l += 1
                    cv2.circle(self.roi, far, 3, [255, 0, 0], -1)

                # draw lines around hand
                cv2.line(self.roi, start, end, [0, 255, 0], 2)

            l += 1

            # print corresponding gestures which are in their ranges
            font = cv2.FONT_HERSHEY_SIMPLEX
            if l == 1:
                if areacnt < 2000:
                    cv2.putText(self.frame, 'Put hand in the box', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    if arearatio < 12:
                        cv2.putText(self.frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    else:
                        cv2.putText(self.frame, '1', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

            elif l == 2:
                cv2.putText(self.frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

            elif l == 3:
                if arearatio < 27:
                    cv2.putText(self.frame, '3', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

            elif l == 4:
                cv2.putText(self.frame, '4', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

            elif l == 5:
                cv2.putText(self.frame, '5', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

            elif l == 6:
                cv2.putText(self.frame, 'reposition', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

            else:
                cv2.putText(self.frame, 'reposition', (10, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        except:
            pass

    def display(self):
        self.fm.show_image(name="Frame", image=self.frame)

    @staticmethod
    def stop(button="q"):
        k = cv2.waitKey(5) & 0xFF
        if k == ord(button):
            return True
        else:
            return False

    def close_all(self):
        self.fm.close_all()
