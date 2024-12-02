
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Inspect the data structure
data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Check if data is inhomogeneous
if all(isinstance(i, (list, np.ndarray)) for i in data):
    # Find the maximum length of the inner sequences
    max_length = max(len(x) for x in data)
    
    # Pad sequences to the maximum length
    padded_data = np.array([np.pad(x, (0, max_length - len(x)), 'constant') for x in data])
else:
    # If not a list of lists or arrays, directly convert
    padded_data = np.asarray(data)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(padded_data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)