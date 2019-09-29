import matplotlib.pyplot as plt
import numpy as np
length = 500
def main():
    X = []
    for i in range(length):
        X.append(np.linspace(0,255, length))
    X = np.asarray(X)
    plt.imshow(X, cmap="gray")
    plt.savefig("grayscale.png")
    plt.show()

if __name__ == "__main__":
    main()