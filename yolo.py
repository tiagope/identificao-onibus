# EXEMNPLO DE CHAMADA
# python yolo.py --imagem images/imagem.jpg --yolo yolo-coco

# importação dos pacotes
import numpy as np
import argparse
import time
import cv2
import os

# Argumentos a serem passados na chamada da aplicação
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagem", required=True,
	help="caminho da imagem")
ap.add_argument("-y", "--yolo", required=True,
	help="caminho para o YOLO treinado")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="probabilidade mínima para considerar na de detecção")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# Carregando as tags do modelo YOLO treinado
labelsPath = os.path.sep.join([args["yolo"], "obj.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# inicializando uma lista de cores para representar cada uma das classes
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# caminhos para o weights e configuração do modelo
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# Carrega o objeto detector do YOLO treinado
print("[INFO] carregando YOLO...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Carrega a sua imagem e calcula as dimensões
image = cv2.imread(args["imagem"])
(H, W) = image.shape[:2]

# determina a camada de saída do YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# contrói um "blob" através da imagem de entrada e executa 
# através do detector de objetos do YOLO, retornando informações das marcacões
# e suas probabilidades
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# exibe informaçÕes do tempo de processamento
print("[INFO] YOLO processamento {:.6f} segundos".format(end - start))

# inicializa a lista de de objetos identificados, marcações, acuraria e ID
boxes = []
confidences = []
classIDs = []

# loop através de cada uma das camadas de saída
for output in layerOutputs:
	# loop através de cada uma das deteções na camada
	for detection in output:
		# extrai o ID da classe e a confiança, ou probabilidade
		# do objeto atualmente identificado
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# filtra as predições fracas.
		# probabilidade deve ser maior que o mínimo desejado
		if confidence > args["confidence"]:			
		#if confidence > args["confidence"] :
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# usar as coordenadas (x, y) para desenhar o topo
			# e o canto esquerdo da marcação
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# atualiza a lista de marcações, confiança e ID das classes			
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)
			
			crop_img = image[y:y+height, x:x+width]			
			cv2.imshow("cropped", crop_img)
			
			

# aplica o intervalo de confiança para não exibir as marcações
# com baixa probabilidade
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

# certifica que existe ao menos uma detecção
if len(idxs) > 0:
	# executa loop através dos index
	for i in idxs.flatten():
		# extrai as coordenadas para montar a marcação
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# desenha uma marcação retangular ao redor do objeto e adiciona 
		# a categoria a acurácia daquela predição	
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

# exibe a imagem
cv2.imshow("Imagem", image)
cv2.waitKey(0)
