import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load Data
data_path = '/mnt/data/Latest_Manufacturing(FOR_EDA).csv'
data = pd.read_csv(data_path)

# Step 2: Data Cleaning
# Removing leading/trailing spaces from column names
data.columns = data.columns.str.strip().str.replace('\s+', '_', regex=True)

# Checking for null values and filling them appropriately
print("Missing Values before handling:")
print(data.isnull().sum())
for col in data.columns:
    if data[col].dtype in ['int64', 'float64']:
        data[col].fillna(data[col].mean(), inplace=True)  # Filling numerical with mean
    else:
        data[col].fillna(data[col].mode()[0], inplace=True)  # Filling categorical with mode

print("Missing Values after handling:")
print(data.isnull().sum())

# Step 3: Feature Selection
selected_features = ['Production_(MT)', 'ENERGY_(Energy_Consumption)', 'TT_TIME_(Total_Cycle_Time_Including_Breakdown)']
data_selected = data[selected_features]

# Step 4: Scaling the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_selected)
data_scaled = pd.DataFrame(data_scaled, columns=selected_features)

# Step 5: Clustering
cluster_range = range(2, 5)  # Trying different cluster sizes
best_k = None
best_score = -1
silhouette_scores = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=42)
    cluster_labels = kmeans.fit_predict(data_scaled)
    score = silhouette_score(data_scaled, cluster_labels)
    silhouette_scores.append(score)
    if score > best_score:
        best_score = score
        best_k = k
        best_model = kmeans

plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title("Silhouette Scores for K-Means Clustering")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()

# Assigning clusters to the dataset
data['Cluster'] = best_model.labels_
data['Cluster'] = data['Cluster'].replace({1: 'Optimal', 0: 'Non-Optimal'})

# Encoding cluster labels for model training
label_encoder = LabelEncoder()
data['Cluster'] = label_encoder.fit_transform(data['Cluster'])

# Step 6: Splitting Data for Model Building
X = data_scaled
y = data['Cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Model Training and Evaluation
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier()
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append([name, accuracy])
    print(f"\n{name} Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Displaying model results in tabular format
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy'])
print(results_df)
