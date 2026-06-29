import numpy as np

class SimpleNeuralNetwork:
    def __init__(self):
        np.random.seed(1)
        self.weights = 2*np.random.randn(3, 1)-1

    def __repr__(self):
        return f"SimpleNeuralNetwork(weights=\n{self.weights})"
    
    #funkcja aktywacji
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def d_sigmoid(self, x):
        return x*(1-x) 
    
    #funkcja propagacji:
    def propagation(self,inputs):
        return self.sigmoid(np.dot(inputs.astype(float), self.weights))
    
    def backpropagation(self, propagation_reusult, train_input, train_output):
        error = train_output - propagation_reusult
        self.weights += np.dot(train_input.T, error * self.d_sigmoid(propagation_reusult))  
    
    def train(self, train_input, train_output,train_iters):
        for _ in range(train_iters):
            propagation_reusult = self.propagation(train_input)
            self.backpropagation(propagation_reusult, train_input, train_output)
            
    
