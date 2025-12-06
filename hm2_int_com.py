import random
import math
import copy
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Config
# ---------------------------
DATA_PATH = r"C:\python_files\Jain.txt"
SEED = 42
EPOCHS = 4000
LEARNING_RATE = 0.1
STANDARDIZE = True
random.seed(SEED)

# ---------------------------
# Improved pure-Python Matrix
# (small, explicit, easy to reason about)
# ---------------------------
class Matrix:
    def __init__(self, data):
        # data: list of rows (each row is a list)
        if not isinstance(data, (list, tuple)):
            raise TypeError("Matrix data must be a list/tuple of rows.")
        if len(data) == 0:
            raise ValueError("Matrix cannot be empty.")
        data = [list(row) for row in data]
        row_lengths = {len(r) for r in data}
        if len(row_lengths) != 1:
            raise ValueError("All rows must have the same number of columns.")
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])
        self.shape = (self.rows, self.cols)

    @classmethod
    def zeros(cls, rows, cols):
        return cls([[0.0 for _ in range(cols)] for _ in range(rows)])

    @classmethod
    def from_flat_col(cls, arr):
        return cls([[float(x)] for x in arr])

    def copy(self):
        return Matrix([row[:] for row in self.data])

    def __repr__(self):
        return f"Matrix({self.data!r})"

    def T(self):
        return Matrix([[self.data[r][c] for r in range(self.rows)] for c in range(self.cols)])

    def transpose(self):
        return self.T()

    def shape_tuple(self):
        return (self.rows, self.cols)

    # elementwise map
    def map(self, fn):
        return Matrix([[fn(self.data[i][j], i, j) for j in range(self.cols)] for i in range(self.rows)])

    # addition, subtraction, hadamard, scalar multiply
    def __add__(self, other):
        if isinstance(other, Matrix):
            assert self.shape == other.shape
            return Matrix([[self.data[i][j] + other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])
        else:
            return Matrix([[self.data[i][j] + other for j in range(self.cols)] for i in range(self.rows)])

    def __sub__(self, other):
        if isinstance(other, Matrix):
            assert self.shape == other.shape
            return Matrix([[self.data[i][j] - other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])
        else:
            return Matrix([[self.data[i][j] - other for j in range(self.cols)] for i in range(self.rows)])

    def __mul__(self, other):
        # scalar multiplication OR hadamard when same shape OR matrix multiply if compatible
        if isinstance(other, (int, float)):
            return Matrix([[self.data[i][j] * other for j in range(self.cols)] for i in range(self.rows)])
        if isinstance(other, Matrix):
            # hadamard if same shape
            if self.shape == other.shape:
                return Matrix([[self.data[i][j] * other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])
            # matrix multiplication
            if self.cols == other.rows:
                out = []
                for i in range(self.rows):
                    row = []
                    for j in range(other.cols):
                        s = 0.0
                        for k in range(self.cols):
                            s += self.data[i][k] * other.data[k][j]
                        row.append(s)
                    out.append(row)
                return Matrix(out)
            raise ValueError("Incompatible shapes for multiplication.")
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    # outer product: self (n x 1), other (1 x m) -> n x m
    def outer(self, other):
        if self.cols != 1 or other.rows != 1:
            raise ValueError("outer() expects self (n x 1) and other (1 x m)")
        return Matrix([[self.data[i][0] * other.data[0][j] for j in range(other.cols)] for i in range(self.rows)])

    def sum(self):
        s = 0.0
        for r in self.data:
            for v in r:
                s += v
        return s

    def to_flat_list(self):
        return [self.data[i][0] for i in range(self.rows)]

# ---------------------------
# Activation functions
# ---------------------------
def sigmoid(mat: Matrix) -> Matrix:
    return mat.map(lambda x, i, j: 1.0 / (1.0 + math.exp(-x)) if -50 < x < 50 else (0.0 if x <= -50 else 1.0))

def sigmoid_derivative_from_activation(a: Matrix) -> Matrix:
    # derivative based on activation value a: a * (1 - a)
    return a * (Matrix([[1.0 for _ in range(a.cols)] for _ in range(a.rows)]) - a)

# ---------------------------
# Data loading and preprocessing
# ---------------------------
def load_jain(path):
    X_rows = []
    y_rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            x1, x2, t = parts
            X_rows.append([float(x1), float(x2)])
            # Note: Class labels 1 and 2 from the dataset are mapped to 1 and 0 internally
            # for binary classification with sigmoid output. This follows standard ML practice.
            y_rows.append(1 if int(t) == 1 else 0)
    return X_rows, y_rows

def standardize_dataset(X_rows):
    cols = len(X_rows[0])
    means = [0.0] * cols
    stds = [0.0] * cols
    n = len(X_rows)
    for j in range(cols):
        means[j] = sum(x[j] for x in X_rows) / n
    for j in range(cols):
        s = sum((x[j] - means[j]) ** 2 for x in X_rows)
        stds[j] = math.sqrt(s / n) if s > 0 else 1.0
        if stds[j] == 0:
            stds[j] = 1.0
    X_scaled = [[(x[j] - means[j]) / stds[j] for j in range(cols)] for x in X_rows]
    return X_scaled, means, stds

def apply_standardization(X_rows, means, stds):
    cols = len(X_rows[0])
    return [[(x[j] - means[j]) / stds[j] for j in range(cols)] for x in X_rows]

def row_to_col_matrix(row):
    return Matrix([[float(row[0])],[float(row[1])]])

# ---------------------------
# Load data
# ---------------------------
print("Loading and preprocessing data...")
X_rows, y_list = load_jain(DATA_PATH)
if len(X_rows) == 0:
    raise RuntimeError("No data loaded. Check DATA_PATH and file format.")

if STANDARDIZE:
    X_rows, MEANS, STDS = standardize_dataset(X_rows)
    print("Data standardized")
else:
    MEANS = None
    STDS = None

combined = list(zip(X_rows, y_list))
random.shuffle(combined)
X_rows, y_list = zip(*combined)
X_rows = list(X_rows)
y_list = list(y_list)

inputs = [row_to_col_matrix(r) for r in X_rows]   # each is 2x1
targets = [Matrix([[v]]) for v in y_list]        # each is 1x1

split = int(0.7 * len(inputs))
train_X, test_X = inputs[:split], inputs[split:]
train_t, test_t = targets[:split], targets[split:]

print(f"Training samples: {len(train_X)}")
print(f"Test samples: {len(test_X)}")

# ---------------------------
# Corrected MLP implementation (vectorized per-sample)
# - Xavier init
# - sigmoid hidden + output
# - correct forward/backprop math
# ---------------------------
class MLP:
    def __init__(self, input_size=2, hidden_size=6, output_size=1, lr=LEARNING_RATE, seed=SEED):
        random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        # Xavier/Glorot initialization
        def xavier(rows, cols):
            limit = math.sqrt(6.0 / (rows + cols))
            return Matrix([[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)])

        # W1: (hidden x input), b1: (hidden x 1)
        self.W1 = xavier(hidden_size, input_size)
        self.b1 = Matrix.zeros(hidden_size, 1)

        # W2: (output x hidden), b2: (output x 1)
        self.W2 = xavier(output_size, hidden_size)
        self.b2 = Matrix.zeros(output_size, 1)

    def feedforward(self, X: Matrix):
        # X: (input x 1)
        self.a1 = (self.W1 * X) + self.b1          # (hidden x 1)
        self.z1 = sigmoid(self.a1)                 # (hidden x 1)
        self.a2 = (self.W2 * self.z1) + self.b2    # (output x 1)
        self.y = sigmoid(self.a2)                  # (output x 1)
        return self.y

    def backprop(self, X: Matrix, t: Matrix):
        # We'll use mean-squared-ish gradient style: delta = (y - t) * sigmoid'(y)
        # (other losses could be used; BCE with sigmoid would simplify delta_o = y - t)
        delta_o = (self.y - t) * sigmoid_derivative_from_activation(self.y)      # (1 x 1)
        # dW2 = delta_o * z1^T  => (output x hidden)
        dW2 = delta_o.outer(self.z1.T())    # delta_o (1x1) outer (1 x hidden) -> (1 x hidden)
        # propagate to hidden: W2^T * delta_o  => (hidden x 1)
        delta_h = (self.W2.T() * delta_o) * sigmoid_derivative_from_activation(self.z1)  # (hidden x 1)
        # dW1 = delta_h * X^T => (hidden x input)
        dW1 = delta_h.outer(X.T())

        # update weights (gradient descent)
        self.W2 = self.W2 - (self.lr * dW2)
        self.b2 = self.b2 - (self.lr * delta_o)

        self.W1 = self.W1 - (self.lr * dW1)
        self.b1 = self.b1 - (self.lr * delta_h)

    def train(self, X_list, t_list, epochs=EPOCHS, verbose=True):
        n = len(X_list)
        for ep in range(epochs):
            idxs = list(range(n))
            random.shuffle(idxs)
            for i in idxs:
                self.feedforward(X_list[i])
                self.backprop(X_list[i], t_list[i])
            if verbose and (ep % 500 == 0 or ep == epochs-1):
                # compute train loss (simple MSE over train set)
                s = 0.0
                for X, t in zip(X_list, t_list):
                    y = self.feedforward(X)
                    diff = y - t
                    s += (diff.data[0][0])**2
                # FIXED: Added the missing print statement
                print(f"Epoch {ep}: Loss = {s/len(X_list):.6f}")

    def predict_proba(self, X: Matrix):
        return float(self.feedforward(X).data[0][0])

    def predict(self, X: Matrix):
        p = self.predict_proba(X)
        return 1 if p >= 0.5 else 0

# ---------------------------
# Train custom network
# ---------------------------
print("\nTraining custom neural network...")
net = MLP(input_size=2, hidden_size=6, output_size=1, lr=LEARNING_RATE, seed=SEED)
net.train(train_X, train_t, epochs=EPOCHS, verbose=True)

# Evaluate on test set
print("\nEvaluating custom neural network...")
correct = 0
for X, t in zip(test_X, test_t):
    pred = net.predict(X)
    true_label = int(t.data[0][0])
    if pred == true_label:
        correct += 1
custom_accuracy = correct / len(test_X)

# ---------------------------
# REQUIRED: Print final weights and biases
# ---------------------------
print("\n" + "="*60)
print("FINAL WEIGHTS AND BIASES")
print("="*60)
print("Input to Hidden Layer Weights (W1):")
for i, row in enumerate(net.W1.data):
    print(f"  Hidden neuron {i}: {[f'{w:.6f}' for w in row]}")
print("\nHidden Layer Biases (b1):")
for i, bias in enumerate(net.b1.data):
    print(f"  Hidden neuron {i}: {bias[0]:.6f}")
print("\nHidden to Output Weights (W2):")
print(f"  Output neuron: {[f'{w:.6f}' for w in net.W2.data[0]]}")
print(f"\nOutput Layer Bias (b2): {net.b2.data[0][0]:.6f}")

# ---------------------------
# Scikit-learn comparison
# ---------------------------
print("\nTraining scikit-learn MLP for comparison...")
X_sklearn = [[x.data[0][0], x.data[1][0]] for x in inputs]
y_sklearn = [t.data[0][0] for t in targets]

split = int(0.7 * len(X_sklearn))
X_train, X_test = X_sklearn[:split], X_sklearn[split:]
y_train, y_test = y_sklearn[:split], y_sklearn[split:]

if STANDARDIZE:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

sklearn_mlp = MLPClassifier(
    hidden_layer_sizes=(6,),
    activation='logistic',  # sigmoid
    learning_rate_init=0.1,
    max_iter=EPOCHS,
    random_state=SEED
)
sklearn_mlp.fit(X_train, y_train)
sklearn_accuracy = sklearn_mlp.score(X_test, y_test)

# ---------------------------
# Final Results Summary
# ---------------------------
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"Custom Neural Network Accuracy: {custom_accuracy:.4f} ({correct}/{len(test_X)} correct)")
print(f"Scikit-learn MLP Accuracy:      {sklearn_accuracy:.4f}")
print(f"Training Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Hidden Layer Size: 6 neurons")
print(f"Activation Function: Sigmoid")
print(f"Train-Test Split: 70%-30%")
print("="*60)

# ---------------------------
# Optional: Visualization
# ---------------------------
def plot_decision_boundary(net, X_test, y_test, title="Decision Boundary"):
    # Create mesh grid
    x_min, x_max = -3, 3
    y_min, y_max = -3, 3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict for each point in mesh
    Z = []
    for x, y in zip(xx.ravel(), yy.ravel()):
        if STANDARDIZE:
            point = Matrix([[(x - MEANS[0]) / STDS[0]], 
                           [(y - MEANS[1]) / STDS[1]]])
        else:
            point = Matrix([[x], [y]])
        pred = net.predict_proba(point)
        Z.append(pred)
    
    Z = np.array(Z).reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu, levels=20)
    plt.colorbar(label='Probability of Class 1')
    
    # Plot test points
    test_points = np.array([[x.data[0][0] for x in X_test], 
                           [x.data[1][0] for x in X_test]]).T
    true_labels = [t.data[0][0] for t in y_test]
    
    scatter = plt.scatter(test_points[:, 0], test_points[:, 1], c=true_labels, 
                           cmap=plt.cm.RdBu, edgecolors='black', s=50)
    plt.legend(handles=scatter.legend_elements()[0], 
               labels=['Class 0', 'Class 1'])
    plt.title(title)
    plt.xlabel('Feature 1 (standardized)' if STANDARDIZE else 'Feature 1')
    plt.ylabel('Feature 2 (standardized)' if STANDARDIZE else 'Feature 2')
    plt.show()

