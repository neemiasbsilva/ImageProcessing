from sklearn import linear_model
import os
import numpy as np
from skimage import io

def openImage(file):
	file_Labels = os.listdir(file)
	np.sort(file_Labels)
	sizeLabels = len(file_Labels)
	X = []
	y = []
	for i in range(0, sizeLabels):
		iteratorLabels_i = os.path.join(file, file_Labels[i])
		fileImage = os.listdir(iteratorLabels_i)
		sizeImage = len(fileImage)
		for j in range(0, sizeImage):
			fileImage = os.path.join(iteratorLabels_i, fileImage[j])
			img = io.imread(fileImage)
			X.append(img)
			y.append(i)

	return X, y


fileTraining = '/home/neemias/Documentos/ComputerScience/procImagem/mnist_png/training'
Xtrain, Ytrain = openImage(fileTraining)

fileTesting = '/home/neemias/Documentos/ComputerScience/procImagem/mnist_png/testing'
Xtesting, Ytesting = openImage(fileTraining)