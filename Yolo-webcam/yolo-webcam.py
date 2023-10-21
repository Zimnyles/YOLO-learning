from ultralytics import YOLO
import cv2
import cvzone
import math
#https://www.youtube.com/watch?v=WgPbbWmnXJ8&t=98s&ab_channel=Murtaza%27sWorkshop-RoboticsandAI# - видео с гайдом
#сначала создадим объект вебкамеры

cap = cv2.VideoCapture(0) #0-id вебкамеры(если их много ставим id>1, если вебкамера одна id=0)
#дальше установим ширину и высоту для камеры
#cap =cv2.VideoCapture(videoURL) - для видеороликов# 
cap.set(3,1280) #Тройка id=3=widht указывает на параметр width=1280
cap.set(4,720) #Четверка id=4=hight указывает на параметр hight=720
#итого разрешение 1280*720
#мы можем установить любое удобное для себя разрешение 

#Используем алгоритм как в файле Yolo-learning/Yolo learning/yolo basics.py
model = YOLO('Yolo-weights/yolov8n.pt')
#results будет в цикле

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

#список классов 
#каждому классу из списка соотвестетствует id (person id=0, bicycle id=1 etc.)


while True:
    success, img = cap.read()
    results = model(img,stream=True) # если Stream=True, то программа будет немного эффективнее работать , чем без  
    #дальше конструкция которая фиксит некоторые ошибки, и позволяет распознать на камере
    for r in results:
        boxes=r.boxes
        for box in boxes:
            
            #Граница объекта
            
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #print(x1, y1, x2, y2)# #получаем фактические значения об объекте которые мы можем использовать в opencv
            #дальше создаем прямоугольник обнаружения
            #cv2.rectangle(img,(x1, y1), (x2, y2), (255,0,255), 3)# #x1y1x2x2-координаты прямоугольника, (255,0,255)-цвет рамки в (r,g,b), цифра 3 - толщина рамки
            w, h = x2-x1, y2-y1 #работаем с помощью разницы координат равные высоте и ширине
            cvzone.cornerRect(img,(x1,y1,w,h)) #в данном случае мы создаем прямоугольник с помощью cvzone
            #дальше нужно получить доверительные значения и имена классов
            
            #Уверенность (вероятность)   
            
            conf = math.ceil((box.conf[0]*100))/100 #умножаем значение на 100 и делим на 100 чтобы получить значение в каноничном виде вероятности (0.xx)
            #дальше нужен небольшой прямоугольник выше основного на котором будет указана инф. об объекте внутри основного прямоуг.
            #нужно использовать функцию которая не позволит вылезти тексту за пределы прямоугольника puttexttrect
            cvzone.putTextRect(img,f'{conf}',(max(0,x1), max(35,y1-20))) #помещаем текст conf в строку f, дальше задаем позицию = стартовое значение наших x и y (y1-20-немного опускаем вниз)
            #но, если мы поднимаем объект за рамки окна программы, то мы не видим значение conf => исправляем с помощью max в значении позиции
            
            #Класс (что за объект)
            
            cls = int(box.cls[0])
            cvzone.putTextRect(img,f'{classNames[cls]}{conf}',(max(0,x1), max(35,y1-20)), scale=1,thickness=3)
            
    cv2.imshow('Image', img )      #Команда выводит изображение
    cv2.waitKey(1)  #устанавливаем задержку 
