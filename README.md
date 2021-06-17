# TensorRT YOLOv4 mask detector model on a Jetson Nano
_Explicación de que consiste este repositorio..._

## Download dataset from Kaggle

Link para la descarga del dataset utilizado...

https://www.kaggle.com/andrewmvd/face-mask-detection

Se ha dividido las imágenes para entrenamiento y test en una relación 80-20%.

## Convert training image labels to YOLO format

```shell
$ cd ${HOME}/project
$ git clone https://github.com/bjornstenger/xml2yolo.git
$ cd xml2yolo
```

Pegar código y explicar los cambios que se han realizado

## Train YOLOv4 on the custom dataset

Comentar que para el entrenamiento se ha utilizado este Notebook que estoy compartiendo. Explicar como se deben de preparar los datos antes de realizar el entrenamiento. Ficheros que se tienen que tener preparados en Google Drive.

https://colab.research.google.com/drive/1MriQiq8z7lxsDWkibTULqymypdeas_d-?usp=sharing

## Convert YOLOv4 to TensorRT model

Explicación de como descargar, instalar librerías necesarias y convertir modelo YOLOv4 que se ha entrenado previamente

```shell
$ cd ${HOME}/project
$ git clone https://github.com/jkjung-avt/tensorrt_demos.git
$ cd tensorrt_demos
```

## Results

Insertar video de YouTube con los resultados expuestos. Hacer pruebas con la mascarilla puesta, quitada y mal puesta. Realizarlo con distintos modelos de mascarillas (distintas formas y colores).

### References

xml2yolo -> https://github.com/bjornstenger/xml2yolo

Darknet -> https://github.com/AlexeyAB/darknet

tensorrt_demos -> https://github.com/jkjung-avt/tensorrt_demos