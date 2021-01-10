import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2



class Detect_Draw_lanes():
    def __init__(self,image):
        self.input_image=image
        self.lines=None
        self.vertices_img=None

    def region_of_interest(self,img):
        """
        Applies an image mask.
        """
        #defining a blank mask
        mask = np.zeros_like(img)   
        #checking number of image channel(color/grayscale) and applying mask
        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255,255,255)
        else:
            ignore_mask_color = 255
        #filling color to pixels inside the polygon    
        cv2.fillPoly(mask, self.vertices_img, ignore_mask_color)
        #image where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        #cv2.imshow('',masked_image)
        return masked_image

    def makeLeftRightline(self):
        """
        find left and right lane coefficients
        """
        left_lines = []
        right_lines = []
        for i in self.lines:
            for x1,y1,x2,y2 in i:
                if x1 == x2:
                    #Vertical Lines
                    pass
                else:
                    m = (y2 - y1) / (x2 - x1)
                    c = y1 - m * x1
                    if m < 0:
                        left_lines.append((m,c))
                    elif m >= 0:
                        right_lines.append((m,c))
        return left_lines,right_lines

    def slope_lines(self,image):
        """
        find mean left and right lane
        """
        img_copy = image.copy()
        
        left_lines,right_lines=self.makeLeftRightline()
        left_line = np.mean(left_lines, axis=0)
        right_line = np.mean(right_lines, axis=0)

        poly_vertices = []
        order = [0,1,3,2]

        for slope, intercept in [left_line, right_line]:
            #getting height of image in y1
            rows, cols = image.shape[:2]
            y1= int(rows) 
            #taking y2 upto 68% of y1
            y2= int(rows*0.68) 
            #y=mx +c can be written as x=(y-c)/m
            x1=int((y1-intercept)/slope)
            x2=int((y2-intercept)/slope)
            poly_vertices.append((x1, y1))
            poly_vertices.append((x2, y2))

            # DRAWING LINES AND PATH ON THE IMAGE
            thickness_of_line=9
            color_of_line=[20, 255, 20]
            lines=np.array([[[x1,y1,x2,y2]]])
            for i in lines:
                for x1,y1,x2,y2 in i:
                    cv2.line(img_copy, (x1, y1), (x2, y2), color_of_line, thickness_of_line)
        poly_vertices = [poly_vertices[i] for i in order]
        #filling polygon color
        cv2.fillPoly(img_copy, pts = np.array([poly_vertices],'int32'), color = (200,20,20))
        final_out=cv2.addWeighted(image,0.7,img_copy,0.4,0.)
        return final_out

    def hough_lines(self,img, rho, theta, threshold, min_line_len, max_line_gap):
        """  
        Returns an image with hough lines drawn.
        """
        self.lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        # Returns a nested list with x1,x2,y1,y2 which are further used to detect the slope and intercept for each line
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        line_img = self.slope_lines(line_img)
        #cv2.imshow(line_img)
        return line_img

    def main(self): 
        #Grayscale
        image=self.input_image
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #Gaussian Smoothing
        smoothed_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        #Canny Edge Detection
        canny_img = cv2.Canny(smoothed_img, 180, 240)

        # defining vertices of image
        rows, cols = image.shape[:2]
        bottom_left  = [cols*0.15, rows]
        top_left     = [cols*0.45, rows*0.6]
        bottom_right = [cols*0.95, rows]
        top_right    = [cols*0.55, rows*0.6] 
        self.vertices_img = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

        #Masked Image Within a Polygon
        masked_img = self.region_of_interest(img = canny_img)
        #Hough Transform Lines
        houghed_lines = self.hough_lines(img = masked_img, rho = 1, theta = np.pi/180, threshold = 40, min_line_len = 20, max_line_gap = 180)
        #Draw lines on edges
        #output= image * 0.8 + houghed_lines * 1. + 0
        output = cv2.addWeighted(image, 0.8, houghed_lines, 1., 0.)
        
        return output