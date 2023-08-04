import pandas as pd
import numpy as np
from qiskit import Aer, transpile, assemble, QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



url                = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns            = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris_data          = pd.read_csv(url, names=columns)

class_mapping      = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
iris_data["class"] = iris_data["class"].map(class_mapping)

X                  = iris_data.drop(columns=["class"]).values
y                  = iris_data["class"].values
X_normalized       = MinMaxScaler().fit_transform(X)
num_qubits         = len(columns) - 1

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Define the quantum feature map for data encoding
zz_feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear')

# Encode the normalized data
encoded_training_data = []
for sample in X_train:
    qc = QuantumCircuit(num_qubits)
    qc.append(zz_feature_map.bind_parameters(sample), range(num_qubits))
    encoded_training_data.append(qc)

encoded_test_data = []
for sample in X_test:
    qc = QuantumCircuit(num_qubits)
    qc.append(zz_feature_map.bind_parameters(sample), range(num_qubits))
    encoded_test_data.append(qc)          

def get_vector_representation(circuits):
    backend      = Aer.get_backend('statevector_simulator')
    qobj         = assemble(transpile(circuits, backend))
    result       = backend.run(qobj).result()
    statevectors = [np.real(result.get_statevector(circuit)) for circuit in circuits]
    return statevectors

# Convert the quantum circuits to a vector representation
X_train_vectors = np.array(get_vector_representation(encoded_training_data))
X_test_vectors  = np.array(get_vector_representation(encoded_test_data))

# Train the Logistic Regression classifier
classifier = LogisticRegression()
classifier.fit(X_train_vectors, y_train)

# Use the trained classifier to predict the labels of the test data
predicted_labels = classifier.predict(X_test_vectors)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, predicted_labels)

print("Classification Accuracy: ",accuracy)
