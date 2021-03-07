# -*- coding: utf-8 -*-
"""
Este script es el archivo principal del programa y se encarga de realizar el analisis del video del Test de Leger,
para llegar a ejecutar este archivo se han realizado previamente los siguientes procesos: 

1. Se realiza la lectura de archivo de puntos (4 PUNTOS) que define la region de interes. En script Seleccion_ROI
2. Se redimensiona la imagen (opcional)
3. Se determina la matriz de transformacion de perspectiva M utilizando como puntos fuente los
   four_points y los puntos destino [(0,H),(W,H),(0,0),(W,0)]. En script Transformacion_Perspectiva
4. Dibujo la polilinea.
5. Mostrar Region aplicando cambio de perspectiva
6. Deteccion de personas en frame, utilizando la funcion deteccion_seguimiento(frame), definida en
   Decetron.py, la cual retorna frame con dibujo de recuadros delimitadores y lista python de
   centroides correspondientes a las detecciones de la clase 'persona'. En script Detectron
7. Dibujar los puntos centroides asociados a las detecciones en vista de pajaro

   NOTA: Para realizar la transformacion de perspectiva a traves de la funcion
         cv2.perspectiveTransform se requiere de:

         1. Matriz M calculada en el paso_3.
         2. Lista de Puntos a transformar en punto flotante 32 como arreglo Numpy:
            src = np.float32(np.array(src_points))
         

"""
from video import FileVideoStream 
from imutils.video import FPS
import cv2
import os
import time
import sys
import pickle
import imutils
import numpy as np
from aux_functions import *
from Detectron import deteccion_seguimiento

# tiempo definido para inicio de prueba

tramo = 1  #contador de repeticiones en cada etapa
#Se inicializan en false el total de etapas a evaluar. 
etapa_1 = False
etapa_2 = False
etapa_3 = False
etapa_4 = False
etapa_5 = False
etapa_6 = False
etapa_7 = False
etapa_8 = False
etapa_9 = False
etapa_10 = False
etapa_11 = False
etapa_12 = False
etapa_13 = False
etapa_14 = False
etapa_15 = False
#----------------------------------------------------------

print("---------------------------------ANALISIS DEPORTIVO DEL TEST DE LEGER----------------------------------")
print("-------------------------------------------------------------------------------------------------------")
print("")
print("El siguiente programa realiza el análisis deportivo del Test de Leger de 15 etapas. Para comenzar con la evaluación por favor ingrese los siguientes parámetros. ")
print("")
video =  input("Ingrese la ruta del video a analizar: ")#'C:/Users/53PW/Desktop/TESIS/Videos/Video.mp4'
salida = input("Ingrese la ruta de salida deseada para el video procesado: ")#'C:/Users/53PW/Desktop/TESIS/Videos/Resultado.avi' 

#----------------------------------------------------------
#ingresar el numero de participantes
n_corredores = 3
from itertools import repeat

list_estados = []

Velocidades = [8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15,15.5]

for i in range(n_corredores) :
    list_estados.append(list(repeat(0, 142)))
    filename = 'Corredor_' + str(i)+'.txt'
    f= open(filename,"w+")
    tex = 'Registro de evaluaciones para el corredor ID: '+ str(i)+'\n'
    f.write(tex+'\n')
    
    f.close() 
    

#---------------------------------------------------------

min_inicio = 0
tramo_analizado = 0 # contra el numero de tramos de la prueba
seg_inicio = int(input("Ingrese el tiempo de inicio del video en segundos: "))
print("")

#-------------------------------------------------
#1. Carga de puntos de calibracion o puntos fuente
if not os.path.isfile('puntos.data'): # el archivo no existe / seleccion no realizada

        print("[INFO] Seleccion no realizada!")
        print("")
        print("La selección de los puntos de calibración no ha sido realizada, para iniciar el analisis del video debe ejecutar primero el archivo Seleccion_ROI.py")
        sys.exit() 
       
else:
        with open('puntos.data', 'rb') as filehandle:
            # read the data as binary data stream
            puntos = pickle.load(filehandle)
            four_points = puntos[0:4]


        

#------------------- FUNCION DE ANALSIS DE POSICION -----------
def analisis_posiciones_1(lista_corredores,Lista_ides,n_etapa,n_tramo):
        global tramo_analizado, n_corredores, Velocidades
        
        print("-------------------------------------------------")
        for i,coord in enumerate(lista_corredores):
                coord_x, coord_y = coord # coordena en y de punto
                                
                # analizar posicion
                if (200 >= coord_y > 0 ) or (1480 >= coord_y >= 1280):
                    print("Corredor ID: ",Lista_ides[i],"APROBADO")    
                    #print('corredor APROBO')
                    VO_MAX = round(((5.857*Velocidades[int(n_etapa)-1])-19.458),2)
                    capaciPul= "su capacidad pulmunar es de: "+ str(VO_MAX)+' VO2max \n'
                    registrar ='Etapa ' + n_etapa +' - ' + ' Tramo ' + str(n_tramo) + '  El corredor ' + ' aprobo y  '+ capaciPul
                else:
                        if coord_y >0 and coord_y <=1480 and i <= n_corredores-1:
                            print("Corredor ID: ",Lista_ides[i],"RECHAZADO")
                            list_estados[Lista_ides[i]][tramo_analizado] = 1  # [[0,0,0],[1,0,0],[0,0,0]]
                            tramos_rechazados = sum(list_estados[Lista_ides[i]])
                            registrar ='Etapa '+ n_etapa +' - '+ ' Tramo ' + str(n_tramo) + '  El corredor  no aprobo\n'
                            if tramos_rechazados >=2:
                                print('El corredor ha sido decalificado: ',i)
                                Descali ='-------------------El corredor fue descalificado en la etapa '+ n_etapa + ' y tramo ' + str(n_tramo) + '-----------------\n'
                                VO_MAX = round(((5.857*Velocidades[int(n_etapa)-1])-19.458),2)
                                filename = 'Corredor_'+str(Lista_ides[i])+'.txt'
                                f= open(filename,"a")
                                capaciPul= "La capacidad pulmunar es de: "+ str(VO_MAX)+' VO2max \n'
                                f.write(Descali+'\n')
                                f.write(capaciPul+'\n')
                                f.close()
                if i <= n_corredores-1 :
                   # registrar la variable registrr EN ARCHIVO .TXT
                   filename = 'Corredor_'+str(Lista_ides[i])+'.txt' 

                   filename1 = 'corredor_'+str(i)+'.txt' 
                   f= open(filename,"a")
                   f.write(registrar+'\n')
                   f.close()
        print("-------------------------------------------------")
        tramo_analizado =  tramo_analizado + 1

#--------------------------------------------
#A través de la función FileVideoStream se inicializa la lectura de los frames del video y con el método
#start se da permiso al buffer de empezar a almacenar esta información.
print ('[INFO] Inicializando video...\n')
print("Analizando Posiciones")
fvs = FileVideoStream(video).start()

#Se inicializa a None el objeto que va a guardar el video que resulte del procesamiento.
writer = None
#Se agrega un delay de 1 segundo al procesamiento para que el buffer empieze a almacenar datos antes de
#entrar al ciclo while.
time.sleep(1.0)

#Se definen los frames por segundos del video utilizando la función de fps importada. 
framepersec = fvs.fps()
#Se redondea la cifra de los frames por segundos a tres cifras significativas. 
frames=round(framepersec,3)

#Se inicializan los contadores a utilizar más adelante para obtener el conteo del tiempo del video. 
minu=0;con=0;cont=0;tiempo=[]

while fvs.more():

    #Se leen los frames que se estan almacenando en el buffer a través del método read y se almacenan en la variable frame. 
    frame = fvs.read()
    #Si no existe un valor para frame el ciclo se sale.  
    if frame is None:
        print("end of the video file...")
        break

    #En el siguiente ciclo if el programa revisa si ya existe un archivo de salida en la ruta suministrada
    #y si no se ha definido el objeto que va a almacenar dicho archivo de salida, de ser asi, se define 
    #el objeto, el cual tiene por parametros de entrada, la ruta donde se va a almacenar el video, el codec
    #que se va a utilizar, los frames por segundo a los que se va a guardar el video, las dimensiones,y un
    #booleano que indica si los frames del video son a color.  
    if salida is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(salida, fourcc, framepersec,
                                 (960, 1100), True)

    #-------------------------------------------------
    # 2. Redimensionar imagen (opcional)
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5,
    #interpolation = cv2.INTER_LINEAR)
    frame_copy = frame.copy()
    frame_copy_1 = frame.copy() # imagen utilizada en la deteccion con YOLOv3
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]    

    #-------------------------------------------------
    # 3. Determinar la matriz de transformacion de perspectiva M 
    M, Minv = get_camera_perspective(frame, puntos[0:4])
    IMAGE_H = frame.shape[0]
    IMAGE_W = frame.shape[1]

    #-------------------------------------------------
    # 4. Dibujar Polilinea 
    pts = np.array([four_points[0], four_points[1], four_points[3], four_points[2]], np.int32)
    thickness=3
    cv2.polylines(frame_copy, [pts], True, (0, 255, 255), thickness, cv2.LINE_AA)
      
    #-------------------------------------------------
    
    # 5. Definimos imagen de perspectiva
    warped_image = cv2.warpPerspective(frame, M, (IMAGE_W, IMAGE_H))
    # redimensionamos resultado para visualizacion
    warped_image = cv2.resize(warped_image,None,fx=0.5, fy=0.5,
                       interpolation = cv2.INTER_LINEAR)

    #-------------------------------------------------
    # 6. Deteccion de personas
    # La funcion pedestrian_detect() retorna la imagen con dibujo de rectangulos delimitadores
    # y centroides, al igual que lista python [(),()...,()] con los centroides asociados
    # a cada deteccion. Estos puntos seran MAPEADOS a la imagen vista de pajaro que se
    # creara en el siguiente paso.
    
    frame, centros, ID_cen =  deteccion_seguimiento(frame_copy_1)
    cv2.polylines(frame, [pts], True, (0, 255, 255), thickness, cv2.LINE_AA)

    #Se inicializan los contador que se van a utilizar para calcular el tiempo del video.
    #El primero se utiliza para llevar la cuenta de los mintutos
    #El segundo se utiliza para llevar la cuenta de los frames que se han procesado. 
    con +=1
    cont +=1
    #-------------------------------------------------
    # 7. Dibujar los puntos centroides asociados a las detecciones en vista de pajaro
    # Los valores de escala definidos (scale_w,scale_h) se definen de manera arbitraria
    # considerando el aspecto de visualizacion que se desea. En nuestro caso utilizaremos
    # un valor de 1.0 para estas 2 variables, es decir que la imagen bird_image tendra las
    # mismas dimensiones utilizdas en el calculo de la matriz M.
    scale_w = 1.0
    scale_h = 1.0
    bird_image = np.zeros((int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8)
    bird_image[:] = (41, 41, 41)
    
    bird_image_1 = np.zeros((int(frame_h * scale_h)+400, int(frame_w * scale_w), 3), np.uint8)
    bird_image_1[:] = (41, 41, 41)
    line_thickness = 2
    cv2.line(bird_image_1, (0, 200), (int(frame_w * scale_w), 200), (0, 0, 255), thickness=line_thickness)
    cv2.line(bird_image_1, (0, int(frame_h * scale_h)+200), (int(frame_w * scale_w), int(frame_h * scale_h)+200), (0, 0, 255), thickness=line_thickness)

    if len(centros)>0: # es decir que hay personas en la escena

            node_radius = 10
            color_node = (192, 133, 156)
            thickness_node = 20

            pts_c = np.float32([centros[:]])
            # realizamos transformacion de perspectiva de toda la lista de centroides
            warped_pt = cv2.perspectiveTransform(pts_c, M)[0]

            lista_corredores = []
            Lista_ides = []

            # recorremos lista de puntos transformados para dibujarlos en la imagen bird_image
            for i in range (len(warped_pt)):
                    warped_pt_scaled = (int(warped_pt[i][0] * scale_w), int(warped_pt[i][1] * scale_h))
                    warped_pt_scaled_1 = (int(warped_pt[i][0] * scale_w), int(warped_pt[i][1] * scale_h)+200)
                    cv2.circle(bird_image, warped_pt_scaled, 4, (0,255,0), 3, cv2.LINE_AA) #coord (x,y)
                    cv2.circle(bird_image_1, warped_pt_scaled_1, 4, (0,255,0), 3, cv2.LINE_AA)
                    lista_corredores.append(warped_pt_scaled_1)
                    Lista_ides.append(ID_cen[i])

    #-----------------------------------------
    # CADENA DE TIEMPO
    #-----------------------------------------
    
    #Se define el tiempo del video. 
    #Se divide el número de frames procesados entre los frames por segundo del video, de esta forma se obtienen
    #los segundos del video.     
    seg=con/frames
    seg_copy = cont/frames
    seg_copy = float(np.round(seg_copy, 3))
    #Si el numero de segundos es mayor a 60 se actualiza el contador de los minutos. 
    if seg >= 60:
        con=0
        minu +=1
    time = f"{minu:02d}:{int(seg):02d}" #Se imprime el tiempo en el formato de min:seg
 
    text2 = 'Tiempo: '+ time
    #Se imprime el tiempo en formato de min:seg en el video de salida. 
    
    if (minu == min_inicio)and (int(seg) == seg_inicio):
        frame[1000:1080,0:1920] = [0,255,0]
        
    if (minu == min_inicio)and (int(seg) == seg_inicio+1):
        etapa_1 = True
        
    seg_etapas = seg_copy - seg_inicio # conteo de segundos para etapas de prueba
    if etapa_1 == True:
        seg_etapa = seg_etapas
        seg_etapa = float(np.round(seg_etapa, 2))

        if (tramo*8.47-0.03) <= seg_etapa <= (tramo*8.47+0.03):
            frame[1000:1080,0:1920] = [0,255,0]
            ## llamar funcion de analisis de posicion de atletas

            analisis_posiciones_1(lista_corredores,Lista_ides,'1',tramo)
            print("Analizando Posiciones")
            tramo = tramo + 1
            
        else:
            frame[1000:1080,0:1920] = [255,255,255]

        if tramo == 8: # se da paso a etapa_2

            tramo = 1
            etapa_1 = False
            etapa_2 = True

        cv2.putText(frame,'Etapa 1', (850,1050), cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
        texto = 'Tramo ' + str(tramo)+ ' ' + 'segundo_ ' + str(seg_etapa) 
        cv2.putText(frame,texto, (1020,1050),cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)

    #----------- ETAPA-2
    if etapa_2 == True and seg_etapas > 60.0:
        seg_etapa = seg_etapas - 60.0
        seg_etapa = float(np.round(seg_etapa, 2))
        
        
        if (tramo*8.00-0.03) <= seg_etapa <= (tramo*8.00+0.03):
            frame[1000:1080,0:1920] = [0,255,0]
            analisis_posiciones_1(lista_corredores,Lista_ides,'2',tramo)
            print("Analizando Posiciones")
            
            tramo = tramo + 1
            
        else:
            frame[1000:1080,0:1920] = [255,255,255]

        if tramo == 8: # se da paso a etapa_3
            tramo = 1
            etapa_2 = False
            etapa_3 = True
            
        cv2.putText(frame,'Etapa 2', (850,1050), cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
        texto = 'Tramo ' + str(tramo)+ ' ' + 'segundo_ ' + str(seg_etapa) 
        cv2.putText(frame,texto, (1020,1050),cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
    #----------- ETAPA-3
    if etapa_3 == True and seg_etapas > 120.0: 
        seg_etapa = seg_etapas - 120.0
        seg_etapa = float(np.round(seg_etapa, 2))
 

        if (tramo*7.58-0.03) <= seg_etapa <= (tramo*7.58+0.03):
            frame[1000:1080,0:1920] = [0,255,0]
            analisis_posiciones_1(lista_corredores,Lista_ides,'3',tramo)
            print("Analizando Posiciones")
            
            tramo = tramo + 1
            
        else:
            frame[1000:1080,0:1920] = [255,255,255]

        if tramo == 8: # se da paso a etapa_4
            tramo = 1
            etapa_3 = False
            etapa_4 = True
        cv2.putText(frame,'Etapa 3', (850,1050), cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
        texto = 'Tramo ' + str(tramo)+ ' ' + 'segundo_ ' + str(seg_etapa) 
        cv2.putText(frame,texto, (1020,1050),cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
    #----------- ETAPA-4
    if etapa_4 == True and seg_etapas > 180.0: 
        seg_etapa = seg_etapas - 180.0
        seg_etapa = float(np.round(seg_etapa, 2))


        if (tramo*7.20-0.03) <= seg_etapa <= (tramo*7.20+0.03):
            frame[1000:1080,0:1920] = [0,255,0]
            analisis_posiciones_1(lista_corredores,Lista_ides,'4',tramo)
            print("Analizando Posiciones")
            
            tramo = tramo + 1
            
        else:
            frame[1000:1080,0:1920] = [255,255,255]

        if tramo == 9: # se da paso a etapa_5
            tramo = 1
            etapa_4 = False
            etapa_5 = True
            
        cv2.putText(frame,'Etapa 4', (850,1050), cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
        texto = 'Tramo ' + str(tramo)+ ' ' + 'segundo_ ' + str(seg_etapa) 
        cv2.putText(frame,texto, (1020,1050),cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
        
    #----------- ETAPA-5
    if etapa_5 == True and seg_etapas > 240.0:
        seg_etapa = seg_etapas - 240.0
        seg_etapa = float(np.round(seg_etapa, 2))


        if (tramo*6.86-0.03) <= seg_etapa <= (tramo*6.86+0.03):
            frame[1000:1080,0:1920] = [0,255,0]
            analisis_posiciones_1(lista_corredores,Lista_ides,'5',tramo)
            print("Analizando Posiciones")
            
            tramo = tramo + 1
            
        else:
            frame[1000:1080,0:1920] = [255,255,255]

        if tramo == 9: # se da paso a etapa_6
            tramo = 1
            etapa_5 = False
            etapa_6 = True
            
        cv2.putText(frame,'Etapa 5', (850,1050), cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
        texto = 'Tramo ' + str(tramo)+ ' ' + 'segundo_ ' + str(seg_etapa) 
        cv2.putText(frame,texto, (1020,1050),cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
    
        #----------- ETAPA-6    
    if etapa_6 == True and seg_etapas > 300.0: 
        seg_etapa = seg_etapas - 300.0
        seg_etapa = float(np.round(seg_etapa, 2))

        if (tramo*6.55-0.03) <= seg_etapa <= (tramo*6.55+0.03):
            frame[1000:1080,0:1920] = [0,255,0]
            analisis_posiciones_1(lista_corredores,Lista_ides,'6',tramo)
            print("Analizando Posiciones")
            tramo = tramo + 1
            
        else:
            frame[1000:1080,0:1920] = [255,255,255]

        if tramo == 10: # se da paso a etapa_7
            tramo = 1
            etapa_6 = False
            etapa_7 = True
            
        cv2.putText(frame,'Etapa 6', (850,1050), cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
        texto = 'Tramo ' + str(tramo)+ ' ' + 'segundo_ ' + str(seg_etapa) 
        cv2.putText(frame,texto, (1020,1050),cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
    #----------- ETAPA-7
    if etapa_7 == True and seg_etapas > 360.0: 
        seg_etapa = seg_etapas - 360.0
        seg_etapa = float(np.round(seg_etapa, 2))

        if (tramo*6.26-0.03) <= seg_etapa <= (tramo*6.26+0.03):
            frame[1000:1080,0:1920] = [0,255,0]
            analisis_posiciones_1(lista_corredores,Lista_ides,'7',tramo)
            print("Analizando Posiciones")
            tramo = tramo + 1
            
        else:
            frame[1000:1080,0:1920] = [255,255,255]

        if tramo == 10: # se da paso a etapa_8
            tramo = 1
            etapa_7 = False
            etapa_8 = True
            
        cv2.putText(frame,'Etapa 7', (850,1050), cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
        texto = 'Tramo ' + str(tramo)+ ' ' + 'segundo_ ' + str(seg_etapa) 
        cv2.putText(frame,texto, (1020,1050),cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
    #----------- ETAPA-8   
    if etapa_8 == True and seg_etapas > 420.0: 
        seg_etapa = seg_etapas - 420.0
        seg_etapa = float(np.round(seg_etapa, 2))

        if (tramo*6.00-0.03) <= seg_etapa <= (tramo*6.00+0.03):
            frame[1000:1080,0:1920] = [0,255,0]
            analisis_posiciones_1(lista_corredores,Lista_ides,'8',tramo)
            print("Analizando Posiciones")
            tramo = tramo + 1
            
        else:
            frame[1000:1080,0:1920] = [255,255,255]

        if tramo == 10: # se da paso a etapa_9
            tramo = 1
            etapa_8 = False
            etapa_9 = True
            
        cv2.putText(frame,'Etapa 8', (850,1050), cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
        texto = 'Tramo ' + str(tramo)+ ' ' + 'segundo_ ' + str(seg_etapa) 
        cv2.putText(frame,texto, (1020,1050),cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
        #----------- ETAPA-9
    if etapa_9 == True and seg_etapas > 480.0: 
        seg_etapa = seg_etapas - 480.0
        seg_etapa = float(np.round(seg_etapa, 2))

        if (tramo*5.76-0.03) <= seg_etapa <= (tramo*5.76+0.03):
            frame[1000:1080,0:1920] = [0,255,0]
            analisis_posiciones_1(lista_corredores,Lista_ides,'9',tramo)
            print("Analizando Posiciones")
            tramo = tramo + 1
            
        else:
            frame[1000:1080,0:1920] = [255,255,255]

        if tramo == 11: # se da paso a etapa_10
            tramo = 1
            etapa_9 = False
            etapa_10 = True
            
        cv2.putText(frame,'Etapa 9', (850,1050), cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
        texto = 'Tramo ' + str(tramo)+ ' ' + 'segundo_ ' + str(seg_etapa) 
        cv2.putText(frame,texto, (1020,1050),cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
    #----------- ETAPA-10
    if etapa_10 == True and seg_etapas > 540.0: 
        seg_etapa = seg_etapas - 540.0
        seg_etapa = float(np.round(seg_etapa, 2))

        if (tramo*5.54-0.03) <= seg_etapa <= (tramo*5.54+0.03):
            frame[1000:1080,0:1920] = [0,255,0]
            analisis_posiciones_1(lista_corredores,Lista_ides,'10',tramo)
            print("Analizando Posiciones")
            tramo = tramo + 1
            
        else:
            frame[1000:1080,0:1920] = [255,255,255]

        if tramo == 11: # se da paso a etapa_11
            tramo = 1
            etapa_10 = False
            etapa_11 = True
            
        cv2.putText(frame,'Etapa 10', (850,1050), cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
        texto = 'Tramo ' + str(tramo)+ ' ' + 'segundo_ ' + str(seg_etapa) 
        cv2.putText(frame,texto, (1020,1050),cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
    #----------- ETAPA-11
    if etapa_11 == True and seg_etapas > 600.0: 
        seg_etapa = seg_etapas - 600.0
        seg_etapa = float(np.round(seg_etapa, 2))

        if (tramo*5.33-0.03) <= seg_etapa <= (tramo*5.33+0.03):
            frame[1000:1080,0:1920] = [0,255,0]
            analisis_posiciones_1(lista_corredores,Lista_ides,'11',tramo)
            print("Analizando Posiciones")
            tramo = tramo + 1
            
        else:
            frame[1000:1080,0:1920] = [255,255,255]

        if tramo == 12: # se da paso a etapa_12
            tramo = 1
            etapa_11 = False
            etapa_12 = True
            
        cv2.putText(frame,'Etapa 11', (850,1050), cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
        texto = 'Tramo ' + str(tramo)+ ' ' + 'segundo_ ' + str(seg_etapa) 
        cv2.putText(frame,texto, (1020,1050),cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
    #----------- ETAPA-12   
    if etapa_12 == True and seg_etapas > 660.0:
        seg_etapa = seg_etapas - 660.0
        seg_etapa = float(np.round(seg_etapa, 2))

        if (tramo*5.14-0.03) <= seg_etapa <= (tramo*5.14+0.03):
            frame[1000:1080,0:1920] = [0,255,0]
            analisis_posiciones_1(lista_corredores,Lista_ides,'12',tramo)
            print("Analizando Posiciones")
            tramo = tramo + 1
            
        else:
            frame[1000:1080,0:1920] = [255,255,255]

        if tramo == 12: # se da paso a etapa_13
            tramo = 1
            etapa_12 = False
            etapa_13 = True
            
        cv2.putText(frame,'Etapa 12', (850,1050), cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
        texto = 'Tramo ' + str(tramo)+ ' ' + 'segundo_ ' + str(seg_etapa) 
        cv2.putText(frame,texto, (1020,1050),cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
    #----------- ETAPA-13
    if etapa_13 == True and seg_etapas > 720.0: 
        seg_etapa = seg_etapas - 720.0
        seg_etapa = float(np.round(seg_etapa, 2))
        if (tramo*4.97-0.03) <= seg_etapa <= (tramo*4.97+0.03):
            frame[1000:1080,0:1920] = [0,255,0]
            analisis_posiciones_1(lista_corredores,Lista_ides,'13',tramo)
            print("Analizando Posiciones")
            tramo = tramo + 1
            
        else:
            frame[1000:1080,0:1920] = [255,255,255]

        if tramo == 13: # se da paso a etapa_4
            tramo = 1
            etapa_13 = False
            etapa_14 = True
        cv2.putText(frame,'Etapa 13', (850,1050), cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
        texto = 'Tramo ' + str(tramo)+ ' ' + 'segundo_ ' + str(seg_etapa) 
        cv2.putText(frame,texto, (1020,1050),cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
    #----------- ETAPA-14   
    if etapa_14 == True and seg_etapas > 780.0: 
        seg_etapa = seg_etapas - 780.0
        seg_etapa = float(np.round(seg_etapa, 2))

        if (tramo*4.80-0.03) <= seg_etapa <= (tramo*4.80+0.03):
            frame[1000:1080,0:1920] = [0,255,0]
            analisis_posiciones_1(lista_corredores,Lista_ides,'14',tramo)
            print("Analizando Posiciones")
            tramo = tramo + 1
            
        else:
            frame[1000:1080,0:1920] = [255,255,255]

        if tramo == 13: # se da paso a etapa_15
            tramo = 1
            etapa_14 = False
            etapa_15 = True
        cv2.putText(frame,'Etapa 14', (850,1050), cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
        texto = 'Tramo ' + str(tramo)+ ' ' + 'segundo_ ' + str(seg_etapa) 
        cv2.putText(frame,texto, (1020,1050),cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
   #----------- ETAPA-15   
    if etapa_15 == True and seg_etapas > 840.0: 
        seg_etapa = seg_etapas - 840.0
        seg_etapa = float(np.round(seg_etapa, 2))

        if (tramo*4.65-0.03) <= seg_etapa <= (tramo*4.65+0.03):
            frame[1000:1080,0:1920] = [0,255,0]
            analisis_posiciones_1(lista_corredores,Lista_ides,'15',tramo)
            print("Analizando Posiciones")
            tramo = tramo + 1
            
        else:
            frame[1000:1080,0:1920] = [255,255,255]

        if tramo == 13:
            tramo = 1
            break
        cv2.putText(frame,'Etapa 15', (850,1050), cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
        texto = 'Tramo ' + str(tramo)+ ' ' + 'segundo_ ' + str(seg_etapa) 
        cv2.putText(frame,texto, (1020,1050),cv2.FONT_HERSHEY_COMPLEX,
                    1.1,(0, 0, 255), 2, lineType=cv2.LINE_AA,)
        
 
    #-------------------------------------------------
    # VISUALIZACION DE RESULTADOS
    #-------------------------------------------------
    cv2.putText(
        frame,
        text2,
        (20,30),
        cv2.FONT_HERSHEY_COMPLEX,
        1.2,
        (0, 0, 255),
        2,
        lineType=cv2.LINE_AA,
        )
    img_1 = cv2.resize(frame_copy, None, fx=0.5, fy=0.5,     # Imagen con region ROI definida
                        interpolation = cv2.INTER_LINEAR)

    img_3 = cv2.resize(frame, None, fx=0.5, fy=0.5,          # Imagen con resultado de YOLOv3
                        interpolation = cv2.INTER_LINEAR)
    
    bird_image = cv2.resize(bird_image,None,fx=0.5, fy=0.5,  # Imagen con vista de pajaro con putnos 
                       interpolation = cv2.INTER_LINEAR)     # con puntos mapeados

    bird_image_2 = bird_image_1.copy()
    bird_image_1 = cv2.resize(bird_image_1,None,fx=0.5, fy=0.5,  # Imagen con vista de pajaro con putnos 
                       interpolation = cv2.INTER_LINEAR) 
    


    #---------------- Resultado final --------------------------

    bird_image_2 = cv2.resize(bird_image_2,(960,560),  # Imagen con vista de pajaro con putnos 
                       interpolation = cv2.INTER_LINEAR)
    resultado_3 = np.vstack((img_3,bird_image_2))
    cv2.imshow("Resultado Final", resultado_3)

    #Si el objeto que esta guardando el video no esta vacio, se procede a escribir sobre el los frames que
    #han sido procesados. 
    if writer is not None:
        writer.write(resultado_3)

    #Se utiliza un key (1) para cerrar la ventana en caso de necesitarlo.   
    key = cv2.waitKey(1) & 0xFF

    # Si se presiona la tecla 'q' se rompe el ciclo. 
    if key == ord('q'):
        break

#Se verifica si se necesita liberar el objeto que esta guardando el video. 
if writer is not None:
    writer.release()

#Limpieza
cv2.destroyAllWindows()

#Se detiene la lectura de frames del video ingresado por linea de comandos. 
fvs.stop()
