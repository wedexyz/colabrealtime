from tkinter import Frame
import numpy as np
import cv2
import time
import torch
import pandas as pd

# Model

model = torch.hub.load( 'yolov5','custom', path='best.pt',source='local') 
cap = cv2.VideoCapture('rec2_Trim2.mp4')


frame_rate = 5
prev = 0
simpen = []

panjang = []
while(cap.isOpened()):
    ret, frame = cap.read()
    #frame = cv2.resize(frame,(600,600))
    
    time_elapsed = time.time() - prev
    if time_elapsed > 1./frame_rate:
        cv2.putText(frame, '', (130, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),1)
        
        prev = time.time()
        detections = model(frame[..., ::-1])
        results = detections.pandas().xyxy[0].to_dict(orient="records")
        
        for result in results:
                con = result['confidence']
                if con > 0.3 :
                    cs  = result['name']
                    x1  = int(result['xmin'])
                    y1  = int(result['ymin'])
                    x2  = int(result['xmax'])
                    y2  = int(result['ymax'])
                    center = int((x1 + x2)/2), int((y1 + y2)/2)

                    #print(simpen)
                    count1 = simpen.count(1)
                    count2 = simpen.count(2)
                    count3 = simpen.count(3)
                    print(count1,count2 ,count3 )
                    
                    cv2.circle(frame, (center), 3, (255, 255, 255), -1) 
                    cv2.rectangle(frame, (x1, y1), (x2, y2),(0, 255, 255),2)
                    cv2.putText(frame, str(cs) , (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255),1)
                    cv2.putText(frame,"x "+str(x2)+" " +"y "+str(y2) , (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 255, 255),1)
                    cv2.putText(frame, str(float(np.around(con, 1))) , (x1+50, y1-5),  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                 
                    print(cs)
                    if cs == 'robot' :
                        r = 1
                        l = y1
                        simpen.append(r)
                        
                        
                    elif cs == 'goalpost' :
                        r = 2
                        l = y1
                        simpen.append(r)
                    elif cs == 'human' :
                        r = 3
                        l = y1
                        simpen.append(r)
                    
                    
        panjang.append([count1,count2,count3])
        cv2.rectangle(frame, (1, -70), (150, 70),(0, 0, 0),-1)
        cv2.putText(frame,"n robot    :" , (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 255, 0),1)
        cv2.putText(frame,"n goalpost :" , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 255, 0),1)
        cv2.putText(frame,"n human    :" , (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 255, 0),1)
        
        cv2.putText(frame, str(count1) , (140, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255),1)
        cv2.putText(frame, str(count2) , (140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255),1)
        cv2.putText(frame, str(count3) , (140, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255),1)

        simpen.clear()
        
        

    if not ret:
        break
    frame = cv2.resize(frame,(800,640))
    
    cv2.imshow('frame', frame)
    total = pd.DataFrame(panjang)
    total.columns = ['robot','goalpost','human']

    print(total)
    print(  "robot",total['robot'].sum(),
            "goalpost",total['goalpost'].sum(),
            "human",total['human'].sum())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

