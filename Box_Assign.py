import cv2
import pickle

width , height = 120,120

try:
    with open("modelList",'rb') as f:
        modellist = pickle.load(f)
except:
    modellist = []

def mouse_click(events,x,y,flags,params):
    if events == cv2.EVENT_LBUTTONDOWN:
        modellist.append((x,y))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i,pos in enumerate(modellist):
            x1,y1=pos
            if x1<x<x1+width and y1<y<y1+height:
                modellist.pop(i)
    
    with open("modelList",'wb') as f:
        pickle.dump(modellist,f)
print(modellist)
while True:
    parking_img = cv2.imread('finalmodel.jpg')
    resized = cv2.resize(parking_img,(1280,720))
    for pos in modellist:
        cv2.rectangle(resized,pos,(pos[0] + width, pos[1] + height),(255,0,255),2)
    cv2.imshow("Your Parking lot",resized)
    cv2.setMouseCallback("Your Parking lot",mouse_click)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break