# YoloV3 custom training video
En este proyecto se presenta el entrenamiento con tu propio dataset para YOLOv3

## Artículo original:
https://pjreddie.com/media/files/papers/YOLOv3.pdf

# Implementación:

### Librerias a descargar

	$pip install opencv-python numpy matplotlib tensorboard terminaltables pillow tqdm
	
	$pup install pytorch torchvision cudatoolkit


### Preparar Dataset
  Ejecutar labelImg.py
  
  	$ cd labelImg
  
  	$ python labelImg.py
  
  Se deben obtener los archivos labels (.txt) de las imagenes de tu dataset
  
  Luego ejecutar el archivo split_train_val.py para generar los archivos train.txt y valid.txt
  
### Preparar archivos de configuración
  Finalmente, son requeridos los siguientes archivos
  
  	En data/custom/images las imagenes del dataset
  
  	En data/custom/labels los labels .txt del dataset
  
  	En data/ los archivos train.txt y valid.txt
  
  	En data/ el archivo classes.names con los nombres de las clases del dataset
  
  	En la carpeta config/ el archivo .cfg que corresponda con el número de clases de nuestro dataset
  
  	En la carpeta config/ modificar el archivo custom.data con el número de clases del dataset
  
  Para el archivo .cfg abrir el archivo con el siguiente comando: 
  
  	$bash create_custom_model.sh <numero de clases>

  Localmente se puede descargar los pesos de la red neuronal para entrenar localmente, con el archivo download_darknet.sh en la carpeta /weights
  	
	$bash download_darknet.sh
	
### Entrenamiento local

	$!python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --epochs 200 --batch_size 4 --pretrained_weights weights/darknet53.conv.74
	
	
### Entrenamiento en google colab
  	$!pip install torch==1.1 torchvision==0.3
  
  	$!pip install opencv-python numpy matplotlib tensorboard terminaltables pillow tqdm
  
  	$!git clone https://github.com/DavidReveloLuna/Yolov3Custom.git
  
  	$cd Yolov3Custom
  
  	$import urllib.request

  	$urllib.request.urlretrieve('https://pjreddie.com/media/files/darknet53.conv.74','/content/Yolov3Custom/weights/darknet53.conv.74')
	
	$from google.colab import drive
        
	$drive.mount('/content/drive')
	
	$!cp -r "/content/drive/My Drive/Desarrollos/YoloCustom/custom" "/content/Yolov3Custom/data"
        
	$!cp -r "/content/drive/My Drive/Desarrollos/YoloCustom/config" "/content/Yolov3Custom"

  Entrar al directorio /usr/local/lib/libpackages/torchvision/transforms/functional.py
  
  Cambiar esta línea
  
	from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
	
  Por esta
  
    from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
    
  Entrenamiento
	
  	$!python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --epochs 200 --batch_size 4 --pretrained_weights weights/darknet53.conv.74
  
### Prueba local en imágen y video

   Descargar el archivo .pth y copiarlo en el directorio checkpoints local
   
   	python detectC.py --image_folder data/samplesC/ --model_def config/yolov3-custom.cfg --weights_path checkpoints/yolov3_ckpt_252.pth --class_path data/custom/classes.names
   
   	python detect_cam.py --model_def config/yolov3-custom.cfg --weights_path checkpoints/yolov3_ckpt_252.pth --class_path data/custom/classes.names --conf_thres 0.6
   
