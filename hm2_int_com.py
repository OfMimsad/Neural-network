# ================================
# Matrix Class (Pure Python)
# ================================
class Matrix:
    def __init__(self, data):
        if not isinstance(data, (list, tuple)):
            raise TypeError("Matrix data must be a list of lists.")
        if len(data) == 0:
            raise ValueError("Matrix cannot be empty.")

        data = [list(row) for row in data]

        row_lengths = {len(row) for row in data}
        if len(row_lengths) != 1:
            raise ValueError("All rows must have the same number of columns.")

        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])
        self.shape = (self.rows, self.cols)

    @classmethod
    def zeros(cls, rows, cols):
        return cls([[0 for _ in range(cols)] for _ in range(rows)])

    @classmethod
    def identity(cls, n):
        m = cls.zeros(n, n)
        for i in range(n):
            m.data[i][i] = 1
        return m

    def __repr__(self):
        return f"Matrix({self.data})"

    def __str__(self):
        return "\n".join(str(row) for row in self.data)

    def copy(self):
        return Matrix([row[:] for row in self.data])

    def transpose(self):
        return Matrix([[self.data[r][c] for r in range(self.rows)]
        for c in range(self.cols)])
    T = property(transpose)

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        return self.data == other.data

    # -------------------
    # Multiplication
    # -------------------
    def __mul__(self, other):
        # Scalar
        if isinstance(other, (int, float)):
            return Matrix([[self.data[i][j] * other
                            for j in range(self.cols)]
                            for i in range(self.rows)])

        # Matrix × Matrix
        if isinstance(other, Matrix):

            # Hadamard
            if self.shape == other.shape:
                return Matrix([[self.data[i][j] * other.data[i][j]
                                for j in range(self.cols)]
                                for i in range(self.rows)])

            # Real matrix multiplication
            if self.cols == other.rows:
                result = []
                for i in range(self.rows):
                    row = []
                    for j in range(other.cols):
                        val = sum(
                            self.data[i][k] * other.data[k][j]
                            for k in range(self.cols)
                        )
                        row.append(val)
                    result.append(row)
                return Matrix(result)

            raise ValueError("Incompatible shapes for multiplication.")

        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    # -------------------
    # Addition / Sub
    # -------------------
    def __add__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Shapes must match for addition.")
        return Matrix([[self.data[i][j] + other.data[i][j]
                        for j in range(self.cols)]
                        for i in range(self.rows)])

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Shapes must match for subtraction.")
        return Matrix([[self.data[i][j] - other.data[i][j]
                        for j in range(self.cols)]
                        for i in range(self.rows)])


import random

eXp = 2.718281828

def sigmoid(x):
    return Matrix([[1 / (1 + eXp**(-x.data[i][j])) 
                    for j in range(x.cols)] 
                    for i in range(x.rows)])

def sigmoid_derivative(x):
    return x * (Matrix([[1 for _ in range(x.cols)] for _ in range(x.rows)]) - x)



# =======================
# Load Dataset
# =======================
inputs = []
targets = []

with open(r"C:\python_files\Jain.txt", "r") as f:
    for line in f:
        x1, x2, t = line.strip().split()
        inputs.append([float(x1), float(x2)])
        targets.append([1 if int(t) == 1 else 0])  # map 1->1, 2->0

# Shuffle
combined = list(zip(inputs, targets))
random.shuffle(combined)
inputs, targets = zip(*combined)

# Convert to matrix
inputs = [Matrix([[x[0]], [x[1]]]) for x in inputs]
targets = [Matrix([[t[0]]]) for t in targets]

# 70/30 split
split = int(0.7 * len(inputs))
train_X, test_X = inputs[:split], inputs[split:]
train_t, test_t = targets[:split], targets[split:]


# =======================
# MLP Neural Network
# =======================
class MLP_NN:
    def __init__(self):
        self.W1 = Matrix([[0.5, 1.2],
                          [0.8, 0.3],
                          [1.1, 0.9],
                          [0.7, 0.4],
                          [-0.3, -0.6],
                          [-0.1, 1.3]])   # (6×2)

        self.b1 = Matrix.zeros(6, 1)

        self.W2 = Matrix([[0.5, -0.3, 0.8, 1.4, -0.7, -0.1]])  # (1×6)
        self.b2 = Matrix.zeros(1, 1)

        self.lr = 0.1

    def feedforward(self, X):
        self.a1 = (self.W1 * X) + self.b1
        self.z1 = sigmoid(self.a1)

        self.a2 = (self.W2 * self.z1) + self.b2
        self.y = sigmoid(self.a2)
        return self.y

    def backprop(self, X, t):
        # output delta
        error = t - self.y
        delta_o = error * sigmoid_derivative(self.y)

        # hidden delta
        hidden_error = self.W2.T * delta_o
        delta_h = hidden_error * sigmoid_derivative(self.z1)

        # update weights
        self.W2 = self.W2 + (self.lr * (delta_o * self.z1.T))
        self.W1 = self.W1 + (self.lr * (delta_h * X.T))

        # update bias
        self.b2 = self.b2 + (self.lr * delta_o)
        self.b1 = self.b1 + (self.lr * delta_h)

    def train(self, X_list, t_list, epochs=2000):
        for _ in range(epochs):
            for X, t in zip(X_list, t_list):
                self.feedforward(X)
                self.backprop(X, t)

    def predict(self, X):
        out = self.feedforward(X).data[0][0]
        return 1 if out < 0.5 else 2


# =======================
# MLP Neural Network
# =======================
class MLP_NN:
    def __init__(self):
        self.W1 = Matrix([[0.5, 1.2],
                          [0.8, 0.3],
                          [1.1, 0.9],
                          [0.7, 0.4],
                          [-0.3, -0.6],
                          [-0.1, 1.3]])   # (6×2)

        self.b1 = Matrix.zeros(6, 1)

        self.W2 = Matrix([[0.5, -0.3, 0.8, 1.4, -0.7, -0.1]])  # (1×6)
        self.b2 = Matrix.zeros(1, 1)

        self.lr = 0.1

    def feedforward(self, X):
        self.a1 = (self.W1 * X) + self.b1
        self.z1 = sigmoid(self.a1)

        self.a2 = (self.W2 * self.z1) + self.b2
        self.y = sigmoid(self.a2)
        return self.y

    def backprop(self, X, t):
        # output delta
        error = t - self.y
        delta_o = error * sigmoid_derivative(self.y)

        # hidden delta
        hidden_error = self.W2.T * delta_o
        delta_h = hidden_error * sigmoid_derivative(self.z1)

        # update weights
        self.W2 = self.W2 + (self.lr * (delta_o * self.z1.T))
        self.W1 = self.W1 + (self.lr * (delta_h * X.T))

        # update bias
        self.b2 = self.b2 + (self.lr * delta_o)
        self.b1 = self.b1 + (self.lr * delta_h)

    def train(self, X_list, t_list, epochs=2000):
        for _ in range(epochs):
            for X, t in zip(X_list, t_list):
                self.feedforward(X)
                self.backprop(X, t)

    def predict(self, X):
        out = self.feedforward(X).data[0][0]
        return 1 if out < 0.5 else 2


nn = MLP_NN()
nn.train(train_X, train_t, epochs=2000)

correct = 0
for X, t in zip(test_X, test_t):
    pred = nn.predict(X)
    true_label = 1 if t.data[0][0] == 1 else 2
    if pred == true_label:
        correct += 1

accuracy = correct / len(test_X)
print("Custom NN Accuracy:", accuracy)



from sklearn.neural_network import MLPClassifier
import numpy as np

# convert back to simple numpy form
X_np = np.array([[x.data[0][0], x.data[1][0]] for x in inputs])
t_np = np.array([1 if t.data[0][0] == 1 else 2 for t in targets])

# split same way
X_train, X_test = X_np[:split], X_np[split:]
t_train, t_test = t_np[:split], t_np[split:]

clf = MLPClassifier(hidden_layer_sizes=(6,),
                    activation='logistic',
                    learning_rate_init=0.1,
                    max_iter=2000)

clf.fit(X_train, t_train)
sk_accuracy = clf.score(X_test, t_test)

print("sklearn NN Accuracy:", sk_accuracy)
