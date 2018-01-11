import numpy as np
import theano
import theano.tensor as T

class SGDRegressor:
    def __init__(self, D):
        print("Using theano SGDRegressor")
        w = np.random.randn(D) / np.sqrt(D)
        self.w = theano.shared(w)
        self.lr = 10e-2

        X = T.matrix('X')
        Y = T.vector('Y')
        Y_hat = X.dot(self.w)
        delta = Y - Y_hat
        cost = delta.dot(delta) # Squared delta
        grad = T.grad(cost, self.w)
        updates = [(self.w, self.w - self.lr * grad)]

        # Define operations
        self.train_op = theano.function(
            inputs = [X, Y],
            updates = updates
        )
        self.predict_op = theano.function(
            inputs = [X],
            outputs = Y_hat
        )

    def partial_fit(self, X, Y):
        self.train_op(X, Y)

    def predict(self, X):
        return self.predict_op(X)
