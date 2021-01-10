import cv2
from Object_detection import detect_ob
from Lane_detection import Detect_Draw_lanes


class run_all():
    def __init__(self):
        self.cap= cv2.VideoCapture('project_video.mp4') 

    def run(self):
        cap=self.cap
        if (cap.isOpened()== False):  
            print("Error opening video  file") 
        while(cap.isOpened()): 
            # Capture frame-by-frame 
            ret, frame = cap.read() 
            if ret: 
                #creating object dection class object
                ob_detect=detect_ob(frame)
                ob_detect.detct_objects()
                bbox=ob_detect.bbox
                label=ob_detect.label
                conf=ob_detect.conf
                #creating draw boxes
                output_image=ob_detect.drawObject_box(frame, bbox, label, conf)
                # Detecting lane lines 
                obj_line=Detect_Draw_lanes(output_image)
                cv2.imshow('frame',obj_line.main())
                # Press any key on keyboard to  exit 
                if cv2.waitKey(25) & 0xFF == ord('q'): 
                    break
            # Break the loop 
            else:  
                break
        cap.release()  
        cv2.destroyAllWindows()




sess=run_all()
sess.run()