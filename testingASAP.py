# # # # Perceptron for AND logic gate
# # # X = [[0,0],[0,1],[1,0],[1,1]]
# # # T = [0,0,0,1]

# # # w1, w2, b = 0, 0, 0
# # # alpha = 1

# # # for epoch in range(10):
# # #     print("Epoch", epoch+1)
# # #     for i in range(len(X)):
# # #         x1, x2 = X[i]
# # #         print("x1 is : ", x1)
# # #         net = w1*x1 + w2*x2 + b

# # #         y = 1 if net >= 0 else 0
# # #         error = T[i] - y
# # #         w1 += alpha * error * x1
# # #         w2 += alpha * error * x2
# # #         b  += alpha * error
# # #         print(f" Input={X[i]} Target={T[i]} Output={y} Error={error}")
# # #     print(f"  w1={w1}, w2={w2}, b={b}\n")

# # # print("Final weights:", w1, w2, "bias:", b)


# # # ADALINE example for learning the AND function
# # X = [[0,0],[0,1],[1,0],[1,1]]
# # T = [0,0,0,1]

# # w1, w2, b = 0, 0, 0
# # alpha = 0.1  # smaller learning rate for stability

# # for epoch in range(20):
# #     total_error = 0
# #     for i in range(len(X)):
# #         x1, x2 = X[i]
# #         net = w1*x1 + w2*x2 + b   # linear output (no step yet)
# #         y = net                    # raw output (ADALINE style)
# #         error = T[i] - y
# #         w1 += alpha * error * x1
# #         w2 += alpha * error * x2
# #         b  += alpha * error
# #         total_error += error**2
# #     print(f"Epoch {epoch+1} -> Weights: {w1:.2f}, {w2:.2f}, Bias: {b:.2f}, Error: {total_error:.3f}")

# # print("Final weights:", w1, w2, "bias:", b)

# # test_X = [[0,0],[0,1],[1,0],[1,1]]



# # for i in range(len(test_X)):
# #     x1,x2 = test_X[i]
# #     net = w1*x1 + w2*x2 + b 
# #     y = 1 if net >= 0 else 0
# #     print(y)

# import matplotlib.pyplot as plt
# import numpy as np

# # ADALINE training visualization
# X = [[0,0],[0,1],[1,0],[1,1]]
# T = [0,0,0,1]

# w1, w2, b = 0.0, 0.0, 0.0
# alpha = 0.1

# print("ADALINE Training Process:")
# print("=" * 50)

# for epoch in range(10):
#     print(f"\nEpoch {epoch + 1}:")
#     for i in range(len(X)):
#         x1, x2 = X[i]
        
#         # ADALINE OUTPUT - NO ACTIVATION!
#         net = w1*x1 + w2*x2 + b
#         linear_output = net  # This IS the output during training
        
#         error = T[i] - linear_output
        
#         print(f"  Input {X[i]}:")
#         print(f"    Linear output (net): {linear_output:.3f}")
#         print(f"    Target: {T[i]}")
#         print(f"    Error: {error:.3f}")
        
#         # Weight updates
#         w1 += alpha * error * x1
#         w2 += alpha * error * x2
#         b += alpha * error

# print(f"\nFinal weights: w1={w1:.3f}, w2={w2:.3f}, b={b:.3f}")

# # Now let's see the decision boundary
# print("\n" + "=" * 50)
# print("TESTING PHASE - Now we add threshold:")
# print("=" * 50)

# for i in range(len(X)):
#     x1, x2 = X[i]
#     net = w1*x1 + w2*x2 + b
    
#     # TRAINING OUTPUT (what ADALINE actually learns)
#     adaline_output = net
    
#     # INFERENCE OUTPUT (what we use for classification)
#     perceptron_output = 1 if net >= 0 else 0
    
#     print(f"Input {X[i]}:")
#     print(f"  ADALINE linear output: {adaline_output:.3f}")
#     print(f"  After threshold: {perceptron_output}")
#     print(f"  Target: {T[i]}")
#     print(f"  Classification: {'CORRECT' if perceptron_output == T[i] else 'WRONG'}")

# x = 0.56
# y = -0.102
# t = 0.3

# print(x*(1-x)*(y*t))


import numpy as np

w1 = np.random.randn(2, 6)
# print(w1)

b1 = np.zeros((1, 6))
print(b1)



# import numpy as np

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def sigmoid_derivative(x):
#     return x * (1 - x)

# class network:
#     def __init__(self, input_num, hidden_num, output_num):
#         self.W1 = np.random.randn(input_num, hidden_num)
#         self.b1 = np.random.randn(1, hidden_num)

#         self.W2 = np.random.randn(hidden_num, output_num)
#         self.b2 = np.zeros((1, output_num))

#         self.lr = 0.1

#     def feedForward(self, X):
#         self.a1 = np.dot(X, self.W1) + self.b1
#         self.z1 = sigmoid(self.a1)

#         self.a2 = np.dot(self.z1, self.W2) + self.b2
#         self.y_out = sigmoid(self.a2)

#         return self.y_out

#     def backpropagation(self, X, t):
#         error = t - self.y_out

#         # output layer delta
#         deltaK = error * sigmoid_derivative(self.y_out)

#         # hidden layer delta
#         hidden_error = np.dot(deltaK, self.W2.T)
#         deltaJ = hidden_error * sigmoid_derivative(self.z1)

#         # weight updates
#         dW_out = self.lr * np.dot(self.z1.T, deltaK)
#         dW_hidden = self.lr * np.dot(X.T, deltaJ)

#         self.W2 += dW_out
#         self.W1 += dW_hidden

#         # bias updates
#         self.b2 += self.lr * np.sum(deltaK, axis=0, keepdims=True)
#         self.b1 += self.lr * np.sum(deltaJ, axis=0, keepdims=True)

#     def train(self, X, t, epochs=1000):
#         for i in range(epochs):
#             self.feedForward(X)
#             self.backpropagation(X, t)
#             if i % 200 == 0:
#                 print(f"Epoch {i}")

# X = np.array([[0,0],[0,1],[1,0],[1,1]])
# t = np.array([[0],[1],[1],[0]])

# nn = network(2, 6, 1)
# nn.train(X, t, epochs=2000)

# print("Final Predictions:")
# print(nn.feedForward(X))
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
