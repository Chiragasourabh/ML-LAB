import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X/np.amax(X,axis=0)
y = y/100
def sigmoid (x):
    return 1/(1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)

epoch=7000 #Setting training iterations

lr=0.1 #Setting learning rate
inputlayer_neurons = 2 #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))
for i in range(epoch):
    h_ip=np.dot(X,wh) + bh
    h_act = sigmoid(h_ip)
    o_ip=np.dot(h_act,wout) + bout
    output = sigmoid(o_ip)
    EO = y-output
    outgrad = derivatives_sigmoid(output)
    d_output = EO* outgrad
    Eh = d_output.dot(wout.T)
    hiddengrad = derivatives_sigmoid(h_act)
    d_hidden = Eh * hiddengrad
    wout += h_act.T.dot(d_output) *lr
    wh += X.T.dot(d_hidden) *lr

print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Learned Output: \n" ,output)
