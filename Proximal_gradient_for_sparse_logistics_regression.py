import numpy as np

def ST(x, t):

    res = np.maximum(0, np.abs(x) - t)
    return np.multiply(np.sign(x),res)

def softmax(x):
    """
    x is a vector (k,1)
    """
    origin_shape = x.shape
    x = x - np.max(x)
    x = np.exp(x)
    x = x/np.sum(x)
    assert x.shape == origin_shape
    return x

def gradient(x, y, Beta):
    """
    x is a vector, array
    y is a number
    Beta is a matrix of parameter
    """
    n_beta = Beta.shape[1]

    #x = x.reshape(1,-1)
    product = np.dot(x, Beta)
    soft = softmax(product)
    soft[y] = soft[y] -1
    x = x.repeat(n_beta).reshape(-1,n_beta)
    beta_gradient = x @ np.diag(soft)

    assert (beta_gradient.shape == Beta.shape)

    return beta_gradient

def proximal(Beta, gradient, t):
    """
    t is stepsize
    """

    mu_matrix = Beta - t*gradient
    prox = ST(mu_matrix, t)
    return prox

def logit_loss(x,y,Beta):
    product = np.dot(x, Beta)
    logit = softmax(product)[y]
    penalty = np.sum(Beta)
    loss = -logit + penalty
    return loss




def proximal_descent(x, y, t, iteration):
    n, p = x.shape[0], x.shape[1]
    k = len(np.unique(y))
    Beta = np.random.randn(p, k)
    loss_v = []
    for i in range(iteration):
        for j in range(n):
            in_x = x[j]
            in_y = y[0][j] 
            grad = gradient(in_x, in_y, Beta)
            Beta = proximal(Beta, grad, t)

            if j%100 == 0:
                loss = logit_loss(in_x, in_y, Beta)
                loss_v.append(loss)
                print(loss) 
    return Beta, loss_v



    


if __name__ == '__main__':
    a = np.array([[1,2],[2,-3]])
    assert (ST(a, 1) == np.array([[0,1], [1,-2]])).all() == True
    assert all(softmax(np.array([1,1])) == np.array([1/2,1/2])) == True
    
    import scipy.io
    ma = scipy.io.loadmat('mnist.mat')
    x = ma['X']
    y = ma['y']
    out_Beta, loss_v = proximal_descent(x, y, 10e-4,5)

    x_test = ma['Xtest']
    y_test = ma['ytest']
    y_pred = np.argmax(x_test @ out_Beta, axis=1)

    print(np.sum(y_test == y_pred))
    precise = np.sum(y_test == y_pred)/len(y_test[0])
    print(precise)