import cv2
import pandas as pd
import numpy as np

def videoToTimeSeries(inpPath, inpRect = [425, 280, 470, 320]): # topleftx toplefty, botrightx, botrighty
    print(inpPath)
    capture = cv2.VideoCapture(inpPath)
    newFrame = 0
    frames = 0 
    rectangles = None
    result = []
   
    
    while (capture.isOpened()):    
        ret , frame = capture.read()            # ret = true or false, frame has a image info
          
        if ret == False:
            break
        newframe  = cv2.resize(frame, (640,480)) 
        if frames ==450:
            rectangles = cv2.rectangle(newframe,(inpRect[0],inpRect[1]),
                                       (inpRect[2],inpRect[3]),(0,255,0),1)
            
            
        
   
        newframe = cv2.GaussianBlur(newframe, (3,3), 10)  # take the low pass filter and blur the image with gaussian
       # newframe = cv2.rectangle(newframe, (inpRect[1], inpRect[3]),(inpRect[0], inpRect[2]), (0,255,0), 1) #draw the rectangle on the image 
        roi = newframe[inpRect[1]:inpRect[3], inpRect[0]:inpRect[2]]
        result.append(np.average(roi))
        frames +=1

    return result,rectangles

df = pd.DataFrame(pd.read_excel( r"C:\Users\BHtosun\Desktop\dataPrediction.xlsx" ,
                                dtype = {'FileName': str, 'Label': str, 'topleftx ': int , 'toplefty ': int,
                                         'botrightx ': int, 'botrighty': int}))

for a in range(len(df)) :
     
    result,rectangle = videoToTimeSeries("FLIR0672.mp4",inpRect= [410,40,490,130])
    
    cv2.imshow("a",rectangle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()