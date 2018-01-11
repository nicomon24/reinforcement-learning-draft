import numpy as np

class SGDRegressor:
    def __init__(self, D):
        print("Using numpy SGDRegressor with eligibility")
        self.w = np.random.randn(D) / np.sqrt(D)

    def partial_fit(self, input_, target, eligibility, lr=10e-3):
      self.w += lr*(target - input_.dot(self.w))*eligibility

    def predict(self, X):
        return X.dot(self.w)
