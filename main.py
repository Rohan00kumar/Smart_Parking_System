
#Module list we import for this process.
import cv2 #importing opencv module.
import pickle #This module is used to store information of available parking slots.
import numpy as np
import serial #This module is used to communicate with arduino uno.
import cvzone
import pytesseract
import imutils
import pandas as pd
import gspread

gc = gspread.service_account(filename='data.json')
sh = gc.open_by_key('1Wn6bEkXA8wR-mHYXfctLRsa8c-HtaNP0Y73-3vUdU6U')

worksheet = sh.sheet1


img_cnt = 1
pwidth,pheight = 1280,1080
#1473,868
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

arduino_board = serial.Serial('COM3',115200)   #here COM3 represents The Port name in Which the Arduino is connected
width , height = 120,120
#video feed
#pcam = cv2.VideoCapture(1)
pcam=cv2.VideoCapture(1) #opencv takes the mentioned video file.
pcam.set(cv2.CAP_PROP_FRAME_WIDTH,pwidth)
pcam.set(cv2.CAP_PROP_FRAME_WIDTH,pheight)
cam = cv2.VideoCapture(0)
with open("modelList",'rb') as f:
        modellist = pickle.load(f)

def avail_parking_space(framepro):
    count_Space = 0
    
    for pos in modellist:
        x,y = pos
        framecrop = framepro[y:y+height,x:x+width]
        count = cv2.countNonZero(framecrop)
        cvzone.putTextRect(frame1,str(count),(x,y+height-10),scale = 1,thickness = 1, offset = 0)
        #Below code is used to indicate free/unoccupied space.
        if count< 1700:
            color = (0,255,0)
            thickness = 5
            count_Space +=1

        #Below code is used to indicate occupied space.    
        else:
            color = (0,0,255)
            thickness = 2
        
        
        cv2.rectangle(frame1,pos,(pos[0] + width, pos[1] + height),color,thickness) #Draw rectangle frame on the parking slots.
        
    cvzone.putTextRect(frame1,f'Free: {count_Space}/{len(modellist)}',(100,50),scale = 4,thickness = 5, offset = 20, colorR=(0,200,0)) #cvzone.putTextRect() is used to display the number of empty slot on the image.
    
    my_num = count_Space

    #comparison section.
    if my_num == 0:
        flag = 0
    
    if my_num > 0:
        flag = 1
    #end of comparision section.

    # arduino code beginning.
    cmd = str(flag) #converting integer into string. 
    myCmd = cmd + '\r' #adding \r at the end of string.
    x = arduino_board.write(bytearray(f"{myCmd}\0","utf-8")) #this line sends commands from python to arduino
    print(myCmd)
    # end of arduino command sending code.

while True:
    #repeatition of the video.
    success, frame1 = pcam.read() #Reading each frame of the clip.
    ret, frame = cam.read()
    framegray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) #convert the colour image into gray image.
    frameblur = cv2.GaussianBlur(framegray,(3,3),1) #blur the gray image.
    frameThreshold = cv2.adaptiveThreshold(frameblur,225,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,16) #convert the blur image into binary image.
    framemedian = cv2.medianBlur(frameThreshold,5) #make the blur image more clear.
    kernel = np.ones((3,3),np.uint8)
    framedialate = cv2.dilate(framemedian,kernel,iterations=1)
    #gate cam.

    if not ret:
        print("fail")
        break
    cv2.imshow("test",frame)
    cv2.resizeWindow("test", 1000, 500)


    k = cv2.waitKey(1)

    if k%265 == 32:
        image = "nump{}.png".format(img_cnt)
        
        img = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 170, 200)
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break
        #print(location)
        mask = np.zeros(gray.shape, np.uint8)
        new_img = cv2.drawContours(mask, [location], 0, 255, -1)
        new_img = cv2.bitwise_and(img, img, mask=mask)
        cv2.namedWindow("Number Plate",cv2.WINDOW_NORMAL)
        cv2.imshow("Number Plate",new_img)

        cv2.imwrite(image,new_img)

        # configuration for tesseract
        config = ('-l eng --oem 1 --psm 3')

        # run tesseract OCR on image
        crop_text = pytesseract.image_to_string(new_img, config=config)
       
        data = pytesseract.image_to_string(new_img)
        # data is stored in CSV file
        #to use this offline uncomment below this
        #raw_data = {'date':[time.asctime( time.localtime(time.time()))],'':[data]}
        # df = pd.DataFrame(raw_data)
        # df.to_csv('data.csv',mode='a')
        user = [data]
        worksheet.insert_row(user)


        # print recognized text
        print(data)

    elif k%256 == 27:
        print("exit")
        break

    
    avail_parking_space(framedialate)
    cv2.imshow('Parking cam',frame1) #this line dispaly the image.
    #gate cam display.
    cv2.imshow('Gate cam',frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
pcam.release()
cam.release()