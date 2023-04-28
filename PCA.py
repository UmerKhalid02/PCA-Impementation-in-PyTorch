import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loading dataset

# I chose Fashion MNIST dataset having 60000 train images and 10000 test images of size 28x28
train_dataset = pd.read_csv('dataset/train.csv')
test_dataset = pd.read_csv('dataset/test.csv')

train_dataset.head()


# labels dictionary

labels = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

x_train = train_dataset.iloc[:, 1:].values
y_train = train_dataset.iloc[:, 0:1].values

x_test = test_dataset.iloc[:, 1:].values
y_test = test_dataset.iloc[: ,0:1].values

index = np.random.randint(0, 60000)
image = x_train[index].reshape((28,28))
label = y_train[index]
plt.imshow(image)
plt.title(labels.get(label[0]))


# KNN class
class KNN:
    def __init__(self, x_train, y_train, k):
        self.k = k
        self.x_train = torch.from_numpy(x_train).to(device)
        self.y_train = torch.from_numpy(y_train).to(device)
    
    def calculate_distances(self, x):
        test_size = x.shape[0]
        train_size = self.x_train.shape[0]
        distances = torch.zeros(test_size, train_size, device=device) # distances array filled with zeros
        for i in range(test_size):
            distances[i] = torch.sqrt(torch.sum((x[i] - self.x_train) ** 2, dim=1)) # euclidean distance
        return distances

    def predict(self, x, y):
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        distances = self.calculate_distances(x) # Compute the distance matrix
        _, indexes = torch.topk(distances, self.k, largest=False) # Find the k-nearest neighbors
        nearest_labels = self.y_train[indexes] # Get the labels of the nearest neighbors
        predictions, _ = torch.mode(nearest_labels, dim=1) # Making predictions

        # calculating probabilities
        test_size = x.shape[0]
        probs = np.zeros((test_size, np.max(self.y_train.cpu().data.numpy())+1))
        for i in range(test_size):
            for j in range(self.k):
                probs[i, nearest_labels[i, j]] += 1
            probs[i] /= self.k

        accuracy = torch.sum(predictions == y).item() / len(y)
        return predictions, probs, accuracy

# PCA

# mean of the training data
mean = np.mean(x_train, axis=0) # axis = 0 will provide us mean of each feature 

# mean subtracting the data
x_train_m = x_train - mean
x_test_m = x_test - mean

# covariance matrix
covariance_matrix = np.cov(x_train_m.T)

# eigenvalues and eigenvectors of the covariance matrix
e_values, e_vectors = np.linalg.eig(covariance_matrix)


# Compute the principal components and find how much information is captured by each principal component

# sorting eigen values along with their corresponding eigen vectors in descending order
sorted_idx = np.argsort(e_values)[::-1] # returns array of indices of same shape
eig_vals_sorted = e_values[sorted_idx] # sorted eigen values in descending order
eig_vecs_sorted = e_vectors[:, sorted_idx] # sorted eigen vectors


total = np.sum(eig_vals_sorted)
variance_explained = [(i / total) for i in eig_vals_sorted]
cumulative_variance_explained = np.cumsum(variance_explained)

# number of components for PCA
n_components = 20

# information captured by n principal components
# calculating variance explained by each component
total_var = np.sum(eig_vals_sorted) 
var_explained = eig_vals_sorted / total_var
print("Information captured by each principal component:\n",variance_explained[:n_components])

# Projecting the data onto the principal components
projection_matrix = eig_vecs_sorted[:, 0:n_components] # selecting first n components
x_train_reduced = np.dot(x_train_m, projection_matrix)
x_test_reduced = np.dot(x_test_m, projection_matrix)

# selecting random test data
index = np.random.randint(0,10000)
test_data = x_test_reduced[index]
test_data = test_data.reshape(1,-1)


knn = KNN(x_train_reduced, y_train, k=3)
y_pred, p, _ = knn.predict(test_data, y_test[index])
print("Predicted: ", labels.get(y_pred.item()), ", Actual: ", labels.get(y_test[index][0]))


# Performance Metrics

y_pred, probs, acc = knn.predict(x_test_reduced, y_test)
y_pred = y_pred.cpu().data.numpy()

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
seaborn.heatmap(confusion_matrix(y_test, y_pred), annot=True,fmt='d')

n_classes = 10
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute the overall ROC curve and AUC
y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
n_classes = y_test_bin.shape[1]
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot the ROC curves
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (AUC = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (AUC = {0:0.2f})'
         ''.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-class Classification')
plt.legend(loc="lower right")
plt.show()