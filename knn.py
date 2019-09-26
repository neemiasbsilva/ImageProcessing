import os
import numpy as np
from skimage import io
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import itertools
def nearestNeighbor(x, Xtrain, y):
	distancia = np.zeros((len(Xtrain)))
	for i in range(0, len(Xtrain)):
		distancia[i] = np.square(np.sum((x - Xtrain[i]))**2)
	minimo = np.argmin(distancia)	
	yprediz = y[minimo]
	
	return yprediz


def openImage(caminho):
	caminho_classes = os.listdir(caminho)
	print caminho_classes
	np.sort(caminho_classes)
	n_classes = len(caminho_classes)
	X = []
	y = []
	for i in range(0, n_classes):
		print('Classe %d de %d'% (i, n_classes))
		caminho_classes_i = os.path.join(caminho, caminho_classes[i])
		caminho_imagens = os.listdir(caminho_classes_i)
		n_imagens = len(caminho_imagens)
		for j in range(0, n_imagens):
			if j% 500 == 0:
				print('\tClasse %d: %d de %d'% (i, j, n_imagens))
			caminho_imagem = os.path.join(caminho_classes_i, caminho_imagens[j])
			img = io.imread(caminho_imagem)
			X.append(img)
			y.append(i)

	return X, y, caminho_classes

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
caminhoTrain = '/home/neemias/Documentos/ComputerScience/procImagem/mnist_png/training'
Xtrain, ytrain, caminho_classes = openImage(caminhoTrain)

Caminho_teste = '/home/neemias/Documentos/ComputerScience/procImagem/mnist_png/testing'
Xteste, yteste, caminho_classes = openImage(Caminho_teste)
Yprediz = np.zeros((len(Xteste)))

for i in range(0,len(Xteste)):
	Yprediz[i] = nearestNeighbor(Xteste[i], Xtrain, ytrain)
	print('Yprediz %d Ytest %d'% (Yprediz[i], yteste[i]) )
count = 0
for i in range(0, len(Xteste)):
	if Yprediz[i] == yteste[i]:
		count += 1

count /= len(Xteste)	

print('Acuracia %f'% np.mean(Yprediz == yteste))
print('Error: %f'% (1-np.mean(Yprediz == yteste)))
print('----Confusion Matrix-----')
print(confusion_matrix(yteste, Yprediz))
cnf_matrix = confusion_matrix(yteste, Yprediz)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=caminho_classes, normalize=True,
                      title='Normalized confusion matrix')
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=caminho_classes,
                      title='Confusion matrix, without normalization')

plt.show()

skplt.metrics.plot_confusion_matrix(yteste, Yprediz, normalize=True)

plt.show()


