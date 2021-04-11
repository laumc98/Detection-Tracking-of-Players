# -*- coding: utf-8 -*-
#Librerias Necesarias

import cv2
from Centroidtracker import CentroidTracker
import detectron2 #Importa todas las funciones y metodos asociados a detectron2.
from detectron2 import model_zoo #Importa la colección de modelos pre-entrenados de Detectron2.
from detectron2.utils.logger import setup_logger #Inicializa el protocolo de registro de eventos de Detectron2.
setup_logger()
from detectron2.engine import DefaultPredictor #Crea un predictor simple con la configuración dada que se ejecuta en un solo dispositivo para una sola imagen de entrada
from detectron2.config import get_cfg #Obtiene una copia de la configuración por defecto de Detectron2
from detectron2.utils.visualizer import Visualizer #Dibuja la información extraida de la detección/segmentació de imagenes.
from detectron2.data import MetadataCatalog #Provee acceso a la metadata de un determinado dataset.
from detectron2.structures import Boxes, BoxMode, pairwise_iou #Soporta las clases relacionadas con los recuadros.

""" En este script se define la configuración de Detectron2, el programa encargado de realizar la detección de los jugadores """
#_______________________________________

#Detectron2 - CentroiTracker

#Se define la configuración para el detector y el centroidtracker.
cfg = get_cfg()#Se crea un objeto que almacena la configuración por defecto de detectron2.
cfg.merge_from_file(model_zoo.get_config_file('COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'
                    ))#Se carga el modelo a utilizar desde la coleccion de modelos entrenados de Detectron2.
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Se define el valor de threshold para  el modelo.
cfg.MODEL.WEIGHTS = \
    model_zoo.get_checkpoint_url('COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'
                                 )#Se carga el modelo entrenado usando la configuraciónn dada.
#cfg.MODEL.DEVICE='cpu'
predictor = DefaultPredictor(cfg)#Se crea el objeto del predictor con la configuración dada.
ct = CentroidTracker()#Se llama a la funcion del cetroidtracker.

def deteccion_seguimiento(frame):
    
    #Ya que Detectron2 acepta imagenes en formato BGR se realiza la conversión del espacio RGB a través
    #de la función cvtColor de cv2.
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #Se llama el objeto del predictor, creado por fuera del ciclo while, y se le pasa como argumento los    
    #frames que se estan leyendo del programa para que haga la detección sobre ellos. Los datos que arroja Detectron2
    #se almacenan en la variable outputs en forma de diccionario.      
    outputs = predictor(frame)

    #Se llama el objeto de salida, instances, el cual almacena los campos de los rectangulos, los keypoints y los scores, entre otros.
    instances = outputs['instances']
    #Se llama el campo de scores, el cual es un Tensor, que almacena en forma de vector los confidence scores,
    #un tipo de threshold que ayuda a filtrar las detecciones erroneas. Ya que el tipo de dato es un tensor 
    #es necesario copiar el tensor a la cpu para luego convertirlo en un array de numpy.  
    scores = instances.scores.cpu().numpy()

    #Se define la variable que va a almacenar las coordendadas de los recuadros para los objetos identificados.
    rects = []

    i = 0
    # Se crea un ciclo for que va a iterar sobre el rango de los objetos detectados. 
    for i in range(0, len(instances)):
        #Se itera sobre los valores de scores, si el valor de dicho threshold es mayor al ingresado vía
        #linea de comando para filtrar las detecciónes erroneas, se obtienen las coordenadas de los rectangulos. 
        confidence = scores[i]
        valor=0.90
        if confidence > valor:

            #Boxes
            #Del objeto de instances se obtienen las coordenadas de los recuadros de los objetos detectados,
            #a través del campo pred_boxes, al igual que para scores,
            #se hace la conversión del tensor a un array numpy.
            boxes = instances.pred_boxes.tensor.cpu().numpy()


            #Keypoints
            #Del objeto de instances se obtienen las coordenadas de los keypoints de los objetos detectados,
            #a través del campo pred_keypoints, y se hace la conversión al array de numpy.
            has_keypoints = instances.has('pred_keypoints')
            if has_keypoints:
                keypoints = instances.pred_keypoints.cpu().numpy()
                #El array que almacena los keypoints tiene tres datos, la coordenada en x, la coordenada en y
                #y la visibilidad de cada punto, por eso, se toman solo los primeros dos datos.
                keypoints = keypoints[:, :, :2]


            #Tracking
            #Se itera sobre las coordenadas de los rectangulos, se asginan a las nuevas variables
            #starx,stary,endx,endy y se cambia el tipo de dato a int. 
            box = boxes[i]
            rects.append(box.astype('int'))#Se almacenan las coordenadas de los rectangulos. 
            (startX, startY, endX, endY) = box.astype('int')
            #se dibujan los rectangulos en el frame. 
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,
                          255, 0), 2)
    
    #Se actualiza el centroidtracker con las coordenadas de los rectangulos que se obtienen cada frame
    objects = ct.update(rects)
    #De objects se obtienen dos listas, la primera corresponde a la lista de los centroides calculados junto
    #a su correspondiente Identificador, y la segunda lista corresponde a las coordenadas de los rectangulos
    #correspondientes a cada identificador. 
    Idcen, Idrec = objects
    centros=[]
    ID_cen = []

    #Se crea un ciclo for que va a iterar sobre los items de la lista de los centroides
    for (objectID, centroid) in list(Idcen.items()):
        #Se dibujan los Id y las coordenadas del centroide sobre el frame que se esta guardando. 
        centros.append(centroid)
        ID_cen.append(objectID)
        text = 'Id{}'.format(objectID)
        cv2.putText(
            frame,
            text,
            (centroid[0] - 20, centroid[1] - 20),
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
            )
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0),
                   -1)

    return frame, centros, ID_cen











    
