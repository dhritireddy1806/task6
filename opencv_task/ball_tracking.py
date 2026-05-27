import cv2
import numpy as np
cap=cv2.VideoCapture("assets/video.mp4")

width=640
height=480
fps=30

dis=lambda a1,b1,a2,b2:((a1-a2)**2+(b1-b2)**2)**0.5

fourcc=cv2.VideoWriter_fourcc(*'mp4v')
out=cv2.VideoWriter('Output.mp4',fourcc,fps,(width,height))

px=None
py=None

while True:
    ret,frame=cap.read()
    if not ret:
        break
    frame=cv2.resize(frame,(width,height))
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    lower=np.array([30,80,80])
    upper=np.array([80, 225,225])
    mask=cv2.inRange(hsv,lower,upper)
    mask=cv2.GaussianBlur(mask,(5,5),0)

    contours,_=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c=max(contours, key=cv2.contourArea)
        (x,y),r=cv2.minEnclosingCircle(c)

        centre=(int(x),int(y))
        radius=int(r)

        if radius>10:
            cv2.circle(frame,centre,radius,(0,255,0),3)
            if px is not None:
                dx=x-px
                dy=y-py
                distance=dis(x,y,px,py)
            px=x
            py=y

    out.write(frame)
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()