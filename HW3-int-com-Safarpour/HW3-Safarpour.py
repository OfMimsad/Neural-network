import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import time
from statistics import mean


def load_txt_dataset(path):
    """Load whitespace OR comma-separated TXT/CSV style datasets."""
    with open(path, 'r') as f:
        first_line = f.readline()
        delimiter = "," if "," in first_line else None  # auto-detect
    
    data = np.loadtxt(path, delimiter=delimiter)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def load_dat_dataset(path):
    """Load KEEL/Weka-style .dat files."""
    data_rows = []
    with open(path, 'r') as f:
        data_section = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("@data"):
                data_section = True
                continue
            if data_section:
                row = [float(x) for x in line.replace(" ", "").split(",")]
                data_rows.append(row)

    data = np.array(data_rows)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y




def preprocess_dataset(X, y):
    unique = np.unique(y)
    label_map = {v: i for i, v in enumerate(unique)}
    y_mapped = np.array([label_map[v] for v in y])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_mapped, test_size=0.30, random_state=42, shuffle=True
    )

    return X_train, X_test, y_train, y_test, scaler, label_map


class MLPModel:
    def __init__(self, hidden_layers=(20,10), learning_rate=0.01, max_iter=500):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            solver='adam',
            activation='logistic'  # sigmoid
        )

    def train_test(self, X_train, y_train, X_test, y_test):
        start = time.time()
        self.model.fit(X_train, y_train)
        end = time.time()

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        runtime = end - start

        return acc, runtime



class RBFNetwork:
    def __init__(self, num_centers=25, sigma=1.0):
        self.num_centers = num_centers
        self.sigma = sigma

    def _rbf(self, X, centers):
        G = np.zeros((X.shape[0], len(centers)))
        for i, c in enumerate(centers):
            G[:, i] = np.exp(-np.linalg.norm(X - c, axis=1)**2 / (2 * self.sigma**2))
        return G

    def train_test(self, X_train, y_train, X_test, y_test):
        start = time.time()

        kmeans = KMeans(n_clusters=self.num_centers, n_init=10).fit(X_train)
        self.centers = kmeans.cluster_centers_

        G_train = self._rbf(X_train, self.centers)
        self.W = np.linalg.pinv(G_train).dot(y_train)

        G_test = self._rbf(X_test, self.centers)
        y_pred = G_test.dot(self.W)

        y_pred = np.rint(y_pred).astype(int)

        end = time.time()
        runtime = end - start
        acc = accuracy_score(y_test, y_pred)

        return acc, runtime



class PNN:
    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.classes = np.unique(y_train)

    def predict(self, X):
        y_pred = []
        for x in X:
            scores = []
            for c in self.classes:
                Xc = self.X_train[self.y_train == c]
                d = np.linalg.norm(Xc - x, axis=1)
                s = np.sum(np.exp(-d**2 / (2 * self.sigma**2)))
                scores.append(s)
            y_pred.append(self.classes[np.argmax(scores)])
        return np.array(y_pred)

    def train_test(self, X_train, y_train, X_test, y_test):
        start = time.time()

        self.train(X_train, y_train)
        y_pred = self.predict(X_test)

        end = time.time()
        runtime = end - start
        acc = accuracy_score(y_test, y_pred)

        return acc, runtime


datasets = {
    "Flame": ("assets/Flame.txt", "txt"),
    "Banana": ("assets/banana.dat", "dat"),
    "Aggregation": ("assets/Aggregation.txt", "txt"),
    "Iris": ("assets/iris.dat", "txt"),
    "Wine": ("assets/wine.dat", "dat")
}



results = {}

for name, (path, ftype) in datasets.items():
    print(f"\n===== Running Dataset: {name} =====")

    if ftype == "txt":
        X, y = load_txt_dataset(path)
    else:
        X, y = load_dat_dataset(path)

    X_train, X_test, y_train, y_test, scaler, label_map = preprocess_dataset(X, y)

    results[name] = {
        "MLP": {"acc": [], "time": []},
        "RBF": {"acc": [], "time": []},
        "PNN": {"acc": [], "time": []}
    }

    for run in range(30):
        print(f"  Run {run+1}/30", end="\r")

        mlp = MLPModel()
        acc, t = mlp.train_test(X_train, y_train, X_test, y_test)
        results[name]["MLP"]["acc"].append(acc)
        results[name]["MLP"]["time"].append(t)

        rbf = RBFNetwork()
        acc, t = rbf.train_test(X_train, y_train, X_test, y_test)
        results[name]["RBF"]["acc"].append(acc)
        results[name]["RBF"]["time"].append(t)

        pnn = PNN()
        acc, t = pnn.train_test(X_train, y_train, X_test, y_test)
        results[name]["PNN"]["acc"].append(acc)
        results[name]["PNN"]["time"].append(t)




print("\n===== FINAL RESULTS =====")
for name in results:
    print(f"\nDataset: {name}")
    print(f"  MLP:  Acc={mean(results[name]['MLP']['acc']):.4f}, Time={mean(results[name]['MLP']['time']):.4f}s")
    print(f"  RBF:  Acc={mean(results[name]['RBF']['acc']):.4f}, Time={mean(results[name]['RBF']['time']):.4f}s")
    print(f"  PNN:  Acc={mean(results[name]['PNN']['acc']):.4f}, Time={mean(results[name]['PNN']['time']):.4f}s")




import pandas as pd

# --------------------------------------------
# Generate MLP Table
# --------------------------------------------
mlp_table = []
for name in results:
    mlp_acc = mean(results[name]["MLP"]["acc"])
    mlp_time = mean(results[name]["MLP"]["time"])
    mlp_table.append([
        name,
        round(mlp_acc, 4),
        X_train.shape[1],       # input neurons
        2,                      # hidden layers
        "(20, 10)",             # hidden neurons
        len(np.unique(y_train)),# output neurons
        0.01,                   # learning rate
        round(mlp_time, 4)
    ])

mlp_df = pd.DataFrame(mlp_table, columns=[
    "Dataset", "Accuracy", "Input Neurons", "Hidden Layers",
    "Hidden Neurons", "Output Neurons", "Learning Rate", "Time (s)"
])

print("\n===== MLP TABLE =====")
print(mlp_df.to_string(index=False))


# --------------------------------------------
# Generate RBF Table
# --------------------------------------------
rbf_table = []
for name in results:
    rbf_acc = mean(results[name]["RBF"]["acc"])
    rbf_time = mean(results[name]["RBF"]["time"])
    rbf_table.append([
        name,
        round(rbf_acc, 4),
        1.0,                # sigma
        round(rbf_time, 4)
    ])

rbf_df = pd.DataFrame(rbf_table, columns=[
    "Dataset", "Accuracy", "Spread (σ)", "Time (s)"
])

print("\n===== RBF TABLE =====")
print(rbf_df.to_string(index=False))


# --------------------------------------------
# Generate PNN Table
# --------------------------------------------
pnn_table = []
for name in results:
    pnn_acc = mean(results[name]["PNN"]["acc"])
    pnn_time = mean(results[name]["PNN"]["time"])
    pnn_table.append([
        name,
        round(pnn_acc, 4),
        0.5,                # sigma
        round(pnn_time, 4)
    ])

pnn_df = pd.DataFrame(pnn_table, columns=[
    "Dataset", "Accuracy", "Spread (σ)", "Time (s)"
])

print("\n===== PNN TABLE =====")
print(pnn_df.to_string(index=False))
