import os

import cv2
from ultralytics import YOLO

model_path = r"runs\detect\train\weights\best.pt"
model = YOLO(model_path)

path = "traffic_dataset/test/images"

if os.path.isdir(path):
    for file in os.listdir(path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")): 
            image_path = os.path.join(path, file)
            
            results = model.predict(image_path, verbose=False)
            
            cizilmis_gorsel = results[0].plot()
            cv2.imshow("Trafik Tespiti", cizilmis_gorsel)
            
            tus = cv2.waitKey(0)
            
            if tus == ord('q'):
                print("Çıkış yapıldı.")
                break
                
    cv2.destroyAllWindows()
else:
    print(f"Belirtilen klasör bulunamadı: {path}")