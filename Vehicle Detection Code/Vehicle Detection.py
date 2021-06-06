
import cv2

cap= cv2.VideoCapture('traffic.mp4')
car_cascade = cv2.CascadeClassifier('vehicle.xml')

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
    cars = car_cascade.detectMultiScale(gray, 1.1, 9)
    for (x,y,w,h) in cars:
            plate = frame[y:y + h, x:x + w]
            cv2.rectangle(frame,(x,y),(x +w, y +h) ,(51 ,51,255),2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (51,51,255), -2)
            cv2.putText(frame, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Vehicle',plate)
    frames = cv2.resize(frame,(1152,700))
    cv2.imshow('Vehicle Detection System', frames)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
