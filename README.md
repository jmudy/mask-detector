# TensorRT YOLOv4 mask detector model on a Jetson Nano
_Explicación de que consiste este repositorio..._

```shell
$ cd ~/

$ mkdir project
$ cd project
```

## Download the dataset

```shell

$ wget https://arcraftimages.s3-accelerate.amazonaws.com/Datasets/Mask/MaskPascalVOC.zip

$ mkdir dataset
$ sudo apt install zip unzip
$ unzip MaskPascalVOC -d dataset/
$ rm MaskPascalVOC.zip
```

Descomprimir en la carpeta project con nombre dataset

```shell
|-- annotations
|   |-- maksssksksss0.xml
|   |-- maksssksksss1.xml
|   |-- maksssksksss2.xml
|   |-- maksssksksss3.xml
|   |-- ...
|-- images
|   |-- maksssksksss0.png
|   |-- maksssksksss1.png
|   |-- maksssksksss2.png
|   |-- maksssksksss3.png
```

## Convert training image labels to YOLO format

```shell
$ cd ~/project

$ git clone https://github.com/jmudy/xml2yolo.git
$ cd xml2yolo
```
Copiar los ficheros de etiquetas en este directorio

```shell
$ cp ../dataset/annotations/*.xml .
$ python3 convert.py
$ rm *.xml
```
## Train YOLOv4 on the custom dataset

Se ha dividido las imágenes para entrenamiento y test en una relación 80-20% respectivamente.

```shell
$ cd ~/project/dataset

$ mkdir obj
$ mkdir test

$ cd images
$ cp $(ls -v | head -n 682) ../obj
$ cp $(ls -v | tail -n 171) ../test

$ cd ../annotations
$ cp $(ls -v | head -n 682) ../obj
$ cp $(ls -v | tail -n 171) ../test

$ zip -r obj.zip obj/
$ zip -r test.zip test/

```

Comentar que para el entrenamiento se ha utilizado este Notebook que estoy compartiendo. Explicar como se deben de preparar los datos antes de realizar el entrenamiento. Ficheros que se tienen que tener preparados en Google Drive.

https://colab.research.google.com/drive/1MriQiq8z7lxsDWkibTULqymypdeas_d-?usp=sharing

## Convert YOLOv4 to TensorRT model

Explicación de como descargar, instalar librerías necesarias y convertir modelo YOLOv4 que se ha entrenado previamente

```shell
$ cd ~/project
$ git clone https://github.com/jkjung-avt/tensorrt_demos.git
$ cd tensorrt_demos
```

## Results

Insertar video de YouTube con los resultados expuestos. Hacer pruebas con la mascarilla puesta, quitada y mal puesta. Realizarlo con distintos modelos de mascarillas (distintas formas y colores).

### References

xml2yolo -> https://github.com/bjornstenger/xml2yolo

Darknet -> https://github.com/AlexeyAB/darknet

tensorrt_demos -> https://github.com/jkjung-avt/tensorrt_demos