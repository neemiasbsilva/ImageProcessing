import os
import numpy as np
from skimage import io
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import itertools


def linearClassifer(X, w, b):
	y = np.dot(X,w)+b
	return np.argmax(y)

def classifierAll(X_text, w, b):
	n = len(X_text)
	y = np.zeros((n,))
	for i in range(n):
		x = X_text[i]
		x = np.reshape(x, (784,))
		y[i] = linearClassifer(x, w, b)
	return y

def openImage(caminho):
	caminho_classes = os.listdir(caminho)
	caminho_classes = np.sort(caminho_classes)
	print caminho_classes
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
			X.append(img / 255.)
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
#caminhoTrain = '/home/neemias/Documentos/ComputerScience/procImagem/mnist_png/training'
#Xtrain, ytrain = openImage(caminhoTrain)

Caminho_teste = '/home/neemias/Documentos/ComputerScience/procImagem/mnist_png/testing'
X_test, yteste, caminho_classes = openImage(Caminho_teste)
print(caminho_classes)
X_test = np.asarray(X_test)
Yprediz = np.zeros((len(X_test)))
w = np.loadtxt('w.txt', delimiter=',')
b = np.loadtxt('b.txt', delimiter=',')

Yprediz = classifierAll(X_test, w, b)

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

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(caminho_classes)):
    fpr[i], tpr[i], _ = roc_curve(yteste[:], Yprediz[:])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(ytest.ravel(), Yprediz.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


