from skimage import io
import os
import numpy as np

def open_images(caminho):
	caminho_classes = os.listdir(caminho)
	caminho_classes = np.sort(caminho_classes)
	X = []
	y = []
	for i in range(0, len(caminho_classes)):
		print('Abrindo imagens: classe %d de %d' % (i, len(caminho_classes)))
		caminho_classe_i = os.path.join(caminho, caminho_classes[i])
		caminho_imagens = os.listdir(caminho_classe_i)
		caminho_imagens = np.sort(caminho_imagens)
		for j in range(0, min(1000,len(caminho_imagens))):
			if j % 500 == 0:
				print('\tclasse %d: %d de %d' % (i, j, len(caminho_imagens)))
			img = io.imread(os.path.join(caminho_classe_i, caminho_imagens[j]))
			#io.imshow(img)
			#io.show()
			w,h = img.shape
			img = (img.reshape((w*h,))) / 255.
			X.append(img)
			y.append(i)
	return X, y

def classificador_linear(x, w, b):



def classifica_todas(X_test, w, b):
	n = len(X_test)
	
	y = np.zeros((n,))
	for i in range(n):
		x = X_test[i]
		y[i] = classificador_linear(x, w, b)
		
	return y

caminho_imagens_test = 'mnist_png/testing'
print('Abrindo imagens')
X_test, Y_test = open_images(caminho_imagens_test)

w = np.loadtxt('w.txt', delimiter=',')
b = np.loadtxt('b.txt', delimiter=',')

y = classifica_todas(X_test, w, b)
print np.mean(Y_test == y)
