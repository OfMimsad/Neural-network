import math
import random

# -------------------------
# Matrix class (pure Python)
# -------------------------
class Matrix:
    def __init__(self, data):
        # accept list/tuple of rows (each row is an iterable)
        if not isinstance(data, (list, tuple)):
            raise TypeError("Matrix data must be a list/tuple of rows.")
        if len(data) == 0:
            raise ValueError("Matrix cannot be empty.")
        # normalize rows to lists
        data = [list(row) for row in data]
        # rectangular check
        row_lengths = {len(r) for r in data}
        if len(row_lengths) != 1:
            raise ValueError("All rows must have the same length.")
        self._data = data
        self._rows = len(data)
        self._cols = len(data[0])

    # properties
    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def shape(self):
        return (self._rows, self._cols)

    # indexing
    def __getitem__(self, pos):
        r, c = pos
        return self._data[r][c]

    def __setitem__(self, pos, value):
        r, c = pos
        self._data[r][c] = value

    # conversions & copy
    def tolist(self):
        return [list(r) for r in self._data]

    def copy(self):
        return Matrix(self.tolist())

    # nice repr/str
    def __repr__(self):
        return f"Matrix({self._data!r})"

    def __str__(self):
        rows = ["[" + ", ".join(f"{x}" for x in row) + "]" for row in self._data]
        return "[\n  " + ",\n  ".join(rows) + "\n]"

    # transpose
    def transpose(self):
        trans = [[self._data[r][c] for r in range(self._rows)] for c in range(self._cols)]
        return Matrix(trans)

    T = property(transpose)

    # equality
    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        return self.tolist() == other.tolist()

    # factory methods
    @classmethod
    def zeros(cls, rows, cols):
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        return cls([[0 for _ in range(cols)] for _ in range(rows)])

    @classmethod
    def identity(cls, n):
        if n <= 0:
            raise ValueError("n must be positive")
        m = cls.zeros(n, n)
        for i in range(n):
            m._data[i][i] = 1
        return m

    @classmethod
    def from_rows(cls, rows_iter):
        return cls([list(r) for r in rows_iter])

    # -------------------------
    # elementwise dot for vectors (helper)
    # -------------------------
    @staticmethod
    def _dot_lists(a, b):
        # both are Python lists of numbers, same length
        return sum(x * y for x, y in zip(a, b))

    # -------------------------
    # operations:
    # __mul__ handles:
    #   - scalar * matrix
    #   - elementwise (Hadamard) if same shape
    #   - matrix multiplication if dims align
    # __rmul__ allows scalar on left
    # __matmul__ handles @ operator (strict matrix multiply)
    # -------------------------
    def __mul__(self, other):
        # scalar multiplication
        if isinstance(other, (int, float)):
            return Matrix([[self._data[i][j] * other for j in range(self._cols)] for i in range(self._rows)])

        # matrix on right
        if isinstance(other, Matrix):
            # hadamard (elementwise)
            if self.shape == other.shape:
                return Matrix([[self._data[i][j] * other._data[i][j] for j in range(self._cols)] for i in range(self._rows)])
            # matrix multiplication (self @ other)
            if self._cols == other._rows:
                m, n, p = self._rows, self._cols, other._cols
                # precompute cols of other for readability
                other_cols = [[other._data[r][c] for r in range(other._rows)] for c in range(other._cols)]
                result = []
                for i in range(m):
                    row_i = self._data[i]
                    result_row = [Matrix._dot_lists(row_i, col_j) for col_j in other_cols]
                    result.append(result_row)
                return Matrix(result)
            raise ValueError("Matrix shapes are incompatible for multiplication.")
        return NotImplemented

    def __rmul__(self, other):
        # scalar * matrix => forward to __mul__
        return self.__mul__(other)

    def __matmul__(self, other):
        # strictly matrix multiplication (no hadamard)
        if not isinstance(other, Matrix):
            return NotImplemented
        if self._cols != other._rows:
            raise ValueError("Inner dimensions must match for matmul.")
        other_cols = [[other._data[r][c] for r in range(other._rows)] for c in range(other._cols)]
        result = []
        for i in range(self._rows):
            row_i = self._data[i]
            result_row = [Matrix._dot_lists(row_i, col_j) for col_j in other_cols]
            result.append(result_row)
        return Matrix(result)

    # addition / subtraction
    def __add__(self, other):
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Shapes must match for addition.")
            return Matrix([[self._data[i][j] + other._data[i][j] for j in range(self._cols)] for i in range(self._rows)])
        # scalar-add not implemented explicitly; user can add Matrix + Matrix of same shape with scalar broadcast if constructed
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Shapes must match for subtraction.")
            return Matrix([[self._data[i][j] - other._data[i][j] for j in range(self._cols)] for i in range(self._rows)])
        return NotImplemented

# -------------------------
# Activation functions (operate elementwise and return Matrix)
# -------------------------
def sigmoid(m: Matrix) -> Matrix:
    # elementwise sigmoid, m can be any shape
    return Matrix([[1.0 / (1.0 + math.exp(-val)) for val in row] for row in m._data])

def sigmoid_derivative_from_activation(a: Matrix) -> Matrix:
    # given activation a = sigmoid(z), derivative is a * (1 - a) elementwise
    return Matrix([[val * (1.0 - val) for val in row] for row in a._data])

# -------------------------
# Utility: convert dataset rows to column-vector Matrices
# -------------------------
def row_to_col_matrix(row):
    # row: iterable of numbers (e.g., [x1, x2])
    return Matrix([[v] for v in row])  # shape (n x 1)

# -------------------------
# Example: load dataset (same format you used)
# each line: x1 x2 t
# -------------------------
def load_dataset(path):
    inputs = []
    targets = []
    with open(path, "r") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) != 3:
                continue
            x1, x2, t_lbl = parts
            inputs.append([float(x1), float(x2)])   # keep as row for now
            targets.append([float(t_lbl)])          # target scalar as 1-element row
    return inputs, targets

# -------------------------
# MLP (2 -> 6 -> 1) sample-by-sample training
# -------------------------
class MLP_NN:
    def __init__(self, input_size=2, hidden_size=6, output_size=1, lr=0.1, seed=None):
        if seed is not None:
            random.seed(seed)
        # initialize weights with small random values
        # W1: hidden_size x input_size
        self.W1 = Matrix([[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)])
        print("***************\n",self.W1)
        # b1: hidden_size x 1
        self.b1 = Matrix([[0.0] for _ in range(hidden_size)])
        # W2: output_size x hidden_size
        self.W2 = Matrix([[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)])
        # b2: output_size x 1
        self.b2 = Matrix([[0.0] for _ in range(output_size)])

        self.lr = lr

    def feedforward(self, x_col: Matrix):
        # x_col: input vector as column Matrix (input_size x 1)
        # z1 = W1 @ x + b1
        self.z1 = (self.W1 @ x_col) + self.b1
        self.a1 = sigmoid(self.z1)               # hidden activation (hidden_size x 1)
        self.z2 = (self.W2 @ self.a1) + self.b2
        self.a2 = sigmoid(self.z2)               # output activation (output_size x 1)
        return self.a2

    def backprop(self, x_col: Matrix, t_col: Matrix):
        # assumes feedforward was called and produced a2, a1, z1, z2
        # output error (a2 - t)
        # delta2 = (a2 - t) * sigmoid'(a2)
        error2 = self.a2 - t_col                 # (output x 1)
        s2 = sigmoid_derivative_from_activation(self.a2)  # (output x 1)
        delta2 = error2 * s2                     # Hadamard (output x 1)

        # gradients for W2: dW2 = delta2 @ a1.T  -> (output x 1) @ (1 x hidden) = output x hidden
        a1T = self.a1.transpose()
        dW2 = delta2 @ a1T                       # uses matmul
        db2 = delta2.copy()                      # output x 1

        # hidden layer error: W2.T @ delta2  -> (hidden x output) @ (output x 1) = hidden x 1
        W2T = self.W2.transpose()
        hidden_error = W2T @ delta2              # (hidden x 1)
        s1 = sigmoid_derivative_from_activation(self.a1)  # (hidden x 1)
        delta1 = hidden_error * s1               # Hadamard (hidden x 1)

        # gradients for W1: dW1 = delta1 @ x.T  -> (hidden x 1) @ (1 x input) = hidden x input
        xT = x_col.transpose()
        dW1 = delta1 @ xT
        db1 = delta1.copy()

        # update weights and biases (gradient descent)
        self.W2 = self.W2 - (self.lr * dW2)
        self.b2 = self.b2 - (self.lr * db2)
        self.W1 = self.W1 - (self.lr * dW1)
        self.b1 = self.b1 - (self.lr * db1)

        # return squared error for monitoring
        # a2 and t_col are matrices 1x1 in typical config; compute scalar loss
        se = 0.0
        for r in range(self.a2.rows):
            for c in range(self.a2.cols):
                diff = self.a2._data[r][c] - t_col._data[r][c]
                se += 0.5 * (diff * diff)
        return se

    def train(self, inputs_list, targets_list, epochs=1000, verbose_every=100):
        # inputs_list: list of rows (e.g., [x1, x2]); targets_list: list of [t]
        n = len(inputs_list)
        for ep in range(1, epochs + 1):
            total_loss = 0.0
            for x_row, t_row in zip(inputs_list, targets_list):
                x_col = row_to_col_matrix(x_row)   # convert to column matrix
                t_col = row_to_col_matrix(t_row)   # target as column matrix
                self.feedforward(x_col)
                loss = self.backprop(x_col, t_col)
                total_loss += loss
            if verbose_every and (ep % verbose_every == 0 or ep == 1 or ep == epochs):
                print(f"Epoch {ep}/{epochs}  total_loss={total_loss:.6f}")
        return total_loss

    def predict(self, x_row):
        x_col = row_to_col_matrix(x_row)
        y = self.feedforward(x_col)
        # return scalar or list depending on shape
        if y.rows == 1 and y.cols == 1:
            return y._data[0][0]
        return y.tolist()

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # adjust this path to your dataset file
    data_path = r"C:\python_files\Jain.txt"

    inputs, targets = load_dataset(data_path)   # inputs: list of [x1,x2], targets: list of [t]

    # quick sanity check: convert one sample to column and print shapes
    if len(inputs) == 0:
        raise RuntimeError("No data loaded; check file path and contents.")

    print("Loaded", len(inputs), "samples.")

    # create network (2 -> 6 -> 1)
    net = MLP_NN(input_size=2, hidden_size=6, output_size=1, lr=0.1, seed=42)

    # train
    net.train(inputs, targets, epochs=2000, verbose_every=200)

    # final predictions on the dataset
    print("\nFinal predictions (first 10):")
    for i, (x_row, t_row) in enumerate(zip(inputs[:10], targets[:10])):
        pred = net.predict(x_row)
        print(f"X={x_row} target={t_row[0]} pred={pred:.4f}")
