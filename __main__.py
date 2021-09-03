from parser_ import parser
from network import neuralnetwork
import matplotlib.pyplot as plt
import sys
import numpy as np
import mnist_process

args = parser.get_parser().parse_args()
args = vars(args)  # Convert argparse Namespace to a dict.
print(args)

x_train,y_train = mnist_process.load_mnist('Mnist')
x_test,y_test = mnist_process.load_mnist('Mnist',kind='t10k')

# Start training!!!
nn = neuralnetwork(args)
nn.fit(x_train[:55000],y_train[:55000],x_train[55000:],y_train[55000:])

# Show the training result
y_test_pred = nn.predict(x_test)
acc = (np.sum(y_test==y_test_pred).astype(float) / x_test.shape[0])
print('\ntraining accuracy: %.2f%%' %(acc*100))

# Help visualize the incorrectly classified picture
wrong_img = x_test[y_test != y_test_pred][:9]
corr_lab = y_test[y_test != y_test_pred][:9]
wrong_lab = y_test_pred[y_test != y_test_pred][:9]
fig, ax = plt.subplots(nrows=3,ncols=3,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(9):
    img = wrong_img[i].reshape(28,28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
    ax[i].set_title('%d.  test:%d/pred:%d' %(i+1,corr_lab[i],wrong_lab[i]))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()