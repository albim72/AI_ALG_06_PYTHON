import numpy as np
from simplenn import SimpleNeuralNetwork


network = SimpleNeuralNetwork()
print(network)

#dane wejściowe
train_inputs = np.array([[1,1,0], [1,1,1], [1,1,0], [1,0,0], [0,1,1], [0,1,0]])
train_outputs = np.array([[1,0,1,1,0,1]]).T
train_iters = 50_000

#trening sieci neuronowej
network.train(train_inputs, train_outputs,train_iters)
print(f"\nwagi po trenngu:\n{network.weights}")

#testowanie
print(f"\nprzykładowe wejście: {train_inputs[0]}")
print(f"przykładowe wyjście: {train_outputs[0]}")
print(f"wyjście po wykonaniu trenningu: {network.propagation(train_inputs[0])}")

#predykcja
test_data = np.array([[1,1,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]])
print(f"\npredykcja z wykorzystania modelu")
for data in test_data:
    print(f"przykładowe wejście: {data}")
    print(f"wyjście po wykonaniu trenningu: {network.propagation(data)}")
