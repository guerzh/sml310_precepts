from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt



from scipy.io import loadmat




#M = loadmat("mnist_all.mat")



def get_test(M):
    batch_xs = np.zeros((0, 28*28))
    batch_y_s = np.zeros( (0, 10))

    test_k =  ["test"+str(i) for i in range(10)]
    for k in range(10):
        batch_xs = np.vstack((batch_xs, ((np.array(M[test_k[k]])[:])/255.)  ))
        one_hot = np.zeros(10)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s,   np.tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s


def get_train(M):
    batch_xs = np.zeros((0, 28*28))
    batch_y_s = np.zeros( (0, 10))

    train_k =  ["train"+str(i) for i in range(10)]
    for k in range(10):
        batch_xs = np.vstack((batch_xs, ((np.array(M[train_k[k]])[:])/255.)  ))
        one_hot = np.zeros(10)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s,   np.tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s


train_x, train_y = get_train(M)
test_x, test_y = get_test(M)



import pandas as pd
from sklearn.preprocessing import LabelEncoder
dat = pd.read_csv("bezdekIris.data", header = None)
all_x = dat.loc[:, 0:3].to_numpy()
all_y_raw = dat.loc[:, 4].to_numpy()

le = preprocessing.LabelEncoder()
le.fit(all_y_raw)
all_y = le.transform(all_y_raw)


dim_x = 4
dim_h = 2
dim_out = 3

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor





idx = np.random.permutation(range(all_x.shape[0]))
train_idx = idx[:60]
test_idx = idx[60:100]
valid_idx = idx[100:]

train_x = all_x[train_idx]
train_y = all_y[train_idx]

test_x = all_x[test_idx]
test_y = all_y[test_idx]

valid_x = all_x[valid_idx]
valid_y = all_y[valid_idx]


x = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
y_classes = Variable(torch.from_numpy(train_y), requires_grad=False).type(dtype_long)



model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),
)

# model = torch.nn.Sequential(
#     torch.nn.Linear(dim_x, dim_out),
# )


loss_fn = torch.nn.CrossEntropyLoss()


learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(1000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y_classes)

    model.zero_grad()  # Zero out the previous gradient computation
    loss.backward()    # Compute the gradient
    optimizer.step()   # Use the gradient information to
                       # make a step


x = Variable(torch.from_numpy(valid_x), requires_grad=False).type(dtype_float)

y_pred = model(x).data.numpy()

np.mean(np.argmax(y_pred, 1) == valid_y)

model[0].weight

model[0].weight.data.numpy()[10, :].shape
plt.imshow(model[0].weight.data.numpy()[12, :].reshape((28, 28)), cmap=plt.cm.jet)


plt.imshow(model[0].weight.data.numpy()[10, :].reshape((28, 28)), cmap=plt.cm.jet)

plt.imshow(model[0].weight.data.numpy()[12, :].reshape((28, 28)), cmap=plt.cm.jet)


my_five = M["train4"][105, :].reshape((28, 28))
plt.imshow(my_five, cmap = plt.cm.gray)
plt.show()