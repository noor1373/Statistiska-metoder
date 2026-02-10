import numpy as np

class LinearRegression:
    def __init__(self):
        self.beta = None
        self.d = None
        self.n = None
        self.sigma_sq = None
    
    def fit(self, X, y):
        """Utför minsta kvadratmetoden"""
        self.n = X.shape[0]
        self.d = X.shape[1]

        ones = np.ones((self.n, 1))
        X_design = np.hstack((ones, X))

        y = y.reshape(-1, 1)

        XT_X_inv = np.linalg.inv(X_design.T @ X_design)
        self.beta = XT_X_inv @ X_design.T @ y

        self.X = X_design
        self.y = y

        self.sigma_sq = self.calculate_variance()

        return self.beta
    
    def predict(self):
        """Beräkna skattade värden (y-hatt)"""
        return self.X @ self.beta
    
    def calculate_variance (self):
        """Beräknar väntevärdesriktigt estimat av variansen """
        predictions = self.predict()
        residuals = self.y - predictions
        sse = np.sum(residuals ** 2)
        variance = sse / (self.n - self.d - 1)
        return variance
    
    def standard_deviation(self):
        """Returnerar standardavvikelsen"""
        return np.sqrt(self.sigma_sq)
    
    def rmse(self):
        """Beräknar Root Mean Squared Error"""
        predictions = self.predict()
        mse = np.mean((self.y - predictions) ** 2)
        return np.sqrt(mse)

