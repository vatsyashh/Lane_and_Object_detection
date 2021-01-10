import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2

class detect_ob():
    def __init__(self,frame):
        self.frame=frame
        self.bbox=None
        self.label=None
        self.conf=None

    def detct_objects(self):
        """
        Detect objects
        """
        self.bbox, self.label, self.conf = cv.detect_common_objects(self.frame)

    def drawObject_box(self,img, bbox, labels, conf):
        """
        Make object draw boxes
        """
        for i, label in enumerate(labels):
            cv2.rectangle(img, (bbox[i][0],bbox[i][1]), (bbox[i][2],bbox[i][3]), (255,255,255), 2)
            cv2.putText(img, label, (bbox[i][0],bbox[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        # output_image = draw_bbox(frame, bbox, label, conf)
        return img