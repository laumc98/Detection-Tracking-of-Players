"""
Se ejecuto en python 3.6

El siguiente script se utiliza para la realizar la transformación de perspectiva de la toma principal del video a analizar.

Para la transformacion de perspectiva:

- Los puntos de origen son:

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
import sys
import pickle
from aux_functions import *  # Funciones
                             # get_camera_perspective()
                             # plot_points_on_bird_eye_view()
                             # plot_pedestrian_boxes_on_image()


#------- Carga de puntos de calibracion ----------
if not os.path.isfile('puntos.data'): # el archivo no existe / seleccion no realizada

        print("[INFO] Seleccion no realizada!")
        sys.exit() 
       
else:
        with open('puntos.data', 'rb') as filehandle:
            # read the data as binary data stream
            puntos = pickle.load(filehandle)
            four_points = puntos[0:4]


#-------------------------------------------------
#Lectura del video a analizar
input_video = "1.mp4"  # 1280x720


cap = cv2.VideoCapture(input_video)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv2.CAP_PROP_FPS))


while cap.isOpened():
   
    ret, frame = cap.read()

    if not ret:
        print("Archivo de video no encontrado...")
        break

    frame_copy = frame.copy()
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]    

    #--------------------------------------------
    # Una vez seleccionado los 4 puntos se define matriz M
    M, Minv = get_camera_perspective(frame, four_points[0:4])
    IMAGE_H = frame.shape[0]
    IMAGE_W = frame.shape[1]
    

    #--------------------------------------------
    # Dibujamos region definida
    pts = np.array([four_points[0], four_points[1], four_points[3], four_points[2]], np.int32)
    thickness=3
    cv2.polylines(frame, [pts], True, (0, 255, 255), thickness, cv2.LINE_AA)

    #--------------------------------------------
    # Definimos imagen de perspectiva
    warped_image = cv2.warpPerspective(frame_copy, M, (IMAGE_W, IMAGE_H))
    # redimensionamos resultado para visualizacion
    warped_image = cv2.resize(warped_image,None,fx=0.5, fy=0.5,
                       interpolation = cv2.INTER_LINEAR)
    cv2.imshow("Bird Image", warped_image )
    cv2.imshow("Frame", frame )
    key = cv2.waitKey(25) & 0xFF
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
