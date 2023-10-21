from ultralytics import YOLO #основные библиотеки
import cv2
model = YOLO('Yolo-weights/yolov8l.pt')
# сверху указан "тип размера" котоый нам скачает библиотека ультралайтс
results = model(r'C:\Users\HUAWEI\Desktop\Yolo-learning\Yolo learning\images\4.png', show=True)
#Сверху полный путь до изображения
cv2.waitKey(0)
#верхняя строчка нужна чтобы изображение не сворачивалось моментально 
