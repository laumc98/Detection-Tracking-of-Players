# -*- coding: utf-8 -*-
"""
Python 3.

El siguiente script se utiliza para seleccionar los 4 puntos que definen la region de interes que, posteriormente
sera usada para realizar la transformacion de perspectiva.

- Los 4 puntos de origen para definicion de ROI:

Punto1 (bl) : abajo a la izquierda
Punto2 (br) : abajo a la derecha
Punto3 (tl) : arriba a la izquierda
Punto4 (tr) : arriba a la derecha

- Los puntos de destino (que deben conservar el mismo orden)
  NO son calculados a partir de los maximos en alto y ancho
  como normalmente se hace.
  Aqui se definen como:

[0, IMAGE_H],          abajo a la izquierda
[IMAGE_W, IMAGE_H],    abajo a la derecha
[0, 0],                arriba a la izquierda
[IMAGE_W, 0]])         arriba a la derecha

"""

import cv2
import os
import numpy as np
import pickle

mouse_pts = []

#Funcion usada para marcar 4 puntos en frame cero
def get_mouse_points(event, x, y, flags, param):
    
    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y

        if len(mouse_pts)<=3:
            cv2.circle(frame, (x, y), 7, (0, 255, 255), 6,cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), 7, (0, 0, 255), 6,cv2.LINE_AA)
        
        mouse_pts.append((x, y))
        print("Punto seleccionado: ")
        print(mouse_pts)

#-------------------------------------------------
#Lectura del video
input_video = "video.mp4"  # 1280x720


cap = cv2.VideoCapture(input_video)
cv2.namedWindow("Seleccion ROI")
cv2.setMouseCallback("Seleccion ROI", get_mouse_points)

frame_num = 0
num_mouse_points = 0
#select = False

while True:
    frame_num += 1
    ret, frame = cap.read()

    if not ret:
        break

    frame_copy = frame.copy()
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    if frame_num == 1:
        while True:
            cv2.imshow("Seleccion ROI", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if len(mouse_pts) == 4:
                cv2.destroyWindow("Seleccion ROI")
                #select = True
                break
        four_points = mouse_pts[0:4]
        #points = mouse_pts[0:6]
        cv2.imwrite('Definicion_ROI.png',frame)

    # Dibujamos region definida
    if len(four_points ) == 4: 
        pts = np.array([four_points[0], four_points[1], four_points[3], four_points[2]], np.int32)
        thickness=3
        cv2.polylines(frame, [pts], True, (0, 255, 255), thickness, cv2.LINE_AA)
    else:
        print('[INFO]No fue realizada seleccion!')
    key = cv2.waitKey(25) & 0xFF
    cv2.imshow("Seleccion ROI", frame)
    if key == 27:
        break
    if key == ord('s'):
        cv2.imwrite('ROI.png',frame)


#---------------------------------------------
#Almacenamiento de la informacion de los puntos seleccionados en un archivo de datos
with open('puntos.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(four_points, filehandle)

print('[INFO]Seleccion guardada!')
cap.release()
cv2.destroyAllWindows()
