# TensorRT YOLOv4 mask detector model on a Jetson Nano
_Underconstruction..._

## Hardware

- [Jetson Nano Developer Kit B01 4GB](https://www.amazon.es/gp/product/B07QWLMR24/ref=ppx_yo_dt_b_asin_title_o03_s01?ie=UTF8&psc=1) (JetPack 4.4)
- [AUKEY Webcam 1080P Full HD](https://www.amazon.es/AUKEY-Linterna-Port%C3%A1til-Ultravioleta-Incorporadas/dp/B01KJZV59K)

## Create a project folder

Run the following command to be in your home directory:
```bash
cd ~/
```

Create a folder called `project`

```bash
mkdir project
cd project
```

## Download the mask dataset

To train the model I used the dataset [Mask Dataset](https://makeml.app/datasets/mask) from MakeML. To download it run the following command:

```bash
wget https://arcraftimages.s3-accelerate.amazonaws.com/Datasets/Mask/MaskPascalVOC.zip
```

Create a folder called dataset to store all images and label files:

```bash
mkdir dataset
#Install the zip and unzip libraries if you don't have them
sudo apt install zip unzip
unzip MaskPascalVOC -d dataset/
rm MaskPascalVOC.zip
```

Inside the dataset folder there are two folders. One with the annotations and the other with the raw images.

```lisp
├── annotations
│   ├── maksssksksss0.xml
│   ├── maksssksksss1.xml
│   ├── maksssksksss2.xml
│   ├── maksssksksss3.xml
│   └── ...
└── images
    ├── maksssksksss0.png
    ├── maksssksksss1.png
    ├── maksssksksss2.png
    ├── maksssksksss3.png
    └── ...
```

## Convert training image labels to YOLO format

```bash
cd ~/project

git clone https://github.com/jmudy/xml2yolo.git
cd xml2yolo
```
Copy the label files to this directory and run the script `convert.py`.

```bash
cp ../dataset/annotations/*.xml .
python3 convert.py
rm *.xml
```
## Train YOLOv4 on the custom dataset

The images for training and test have been divided in a ratio of 80-20% respectively.

```bash
cd ~/project

git clone https://github.com/jmudy/mask-detector
cp -r mask-detector/yolov4-mask/ .
rm -r -f mask-detector/
```

```bash
cd ~/project/dataset

mkdir obj
mkdir test

cd images
cp $(ls -v | head -n 682) ../obj
cp $(ls -v | tail -n 171) ../test

cd ../annotations
cp $(ls -v | head -n 682) ../obj
cp $(ls -v | tail -n 171) ../test

cd ..
```

```bash
zip -r obj.zip obj/ ../yolov4-mask/
zip -r test.zip test/ ../yolov4-mask/
```

Copiar la carpeta `yolov4-mask` a la raíz de tu carpeta Google Drive.

Comentar que para el entrenamiento se ha utilizado este Notebook que estoy compartiendo. Explicar como se deben de preparar los datos antes de realizar el entrenamiento. Ficheros que se tienen que tener preparados en Google Drive.

https://colab.research.google.com/drive/1MriQiq8z7lxsDWkibTULqymypdeas_d-?usp=sharing

Al terminar el entrenamiento cambiar nombre del fichero `/mydrive/yolov4-mask/backup/yolov4-mask_best.weights` a `yolov4-mask.weights`.

## Convert YOLOv4 to TensorRT model


```bash
cd ~/project
git clone https://github.com/jkjung-avt/tensorrt_demos.git
cd tensorrt_demos
```

```bash
cd ~/project/tensorrt_demos/ssd
./install_pycuda.sh
```

```bash
sudo apt-get install protobuf-compiler libprotoc-dev
sudo pip3 install onnx==1.4.1
```

```bash
cd ~/project/tensorrt_demos/plugins
make
```

Copiar en esta carpeta los ficheros yolov4-mask.cfg y yolov4-mask.weights

```bash
cd ../yolo
python3 yolo_to_onnx.py -m yolov4-mask
python3 onnx_to_tensorrt.py -m yolov4-mask
```

Cambiar el fichero `~/project/tensorrt_demos/utils/yolo_classes.py`:

```bash
"""yolo_classes.py

NOTE: Number of YOLO COCO output classes differs from SSD COCO models.
"""

COCO_CLASSES_LIST = [
    'with_mask',
    'without_mask',
    'mask_weared_incorrect',
]

# For translating YOLO class ids (0~79) to SSD class ids (0~90)
yolo_cls_to_ssd = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]

def get_cls_dict(category_num):
    """Get the class ID to name translation dictionary."""
    if category_num == 3:
        return {i: n for i, n in enumerate(COCO_CLASSES_LIST)}
    else:
        return {i: 'CLS%d' % i for i in range(category_num)}
```

## Results

<p align="center">
    <img  width="425" src="gif/result.gif">
</p>

Cambiar que por defecto se detectan 3 clases y aumentar el umbral de confianza:

```bash
sed -i '33s/default=80/default=3/' trt_yolo.py
sed -i '101s/conf_th=0.3/conf_th=0.8/' trt_yolo.py
```
Ejecutar el siguiente comando para visualizar los resultados:
```bash
cd ~/project/tensorrt_demos
python3 trt_yolo.py --usb 0 -m yolov4-mask
```

Insertar video de YouTube con los resultados expuestos. Hacer pruebas con la mascarilla puesta, quitada y mal puesta. Realizarlo con distintos modelos de mascarillas (distintas formas y colores).

### References  

   This project is totally inspired by the following previous repositories:

  * [xml2yolo](https://github.com/bjornstenger/xml2yolo)
  * [darknet](https://github.com/AlexeyAB/darknet)
  * [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos)
