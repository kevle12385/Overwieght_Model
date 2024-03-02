import pandas as pd
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score


# Preset matplotlib figure sizes.
matplotlib.rcParams['figure.figsize'] = [9, 6]

# To make the results reproducible, set the random seed value.
np.random.seed(42)

data = pd.read_csv("ObesityDataSet.csv")
le = preprocessing.LabelEncoder() # it preserves the ordinal relationship by assigning numerical labels in a way that reflects the order of the categories


Gender = le.fit_transform(list(data["Gender"])) # get entire buying column and turn them into a list and transform data into int values
Age = le.fit_transform(list(data["Age"])) 
Height = le.fit_transform(list(data["Height"])) 
Weight = le.fit_transform(list(data["Weight"])) 
family_history_with_overweight = le.fit_transform(list(data["family_history_with_overweight"])) 
FAVC = le.fit_transform(list(data["FAVC"])) #Frequent consumption of high caloric food
FCVC = le.fit_transform(list(data["FCVC"])) #Frequency of consumption of vegetables
NCP = le.fit_transform(list(data["NCP"])) #Number of main meals
CAEC = le.fit_transform(list(data["CAEC"])) #Consumption of food between meals
SMOKE = le.fit_transform(list(data["SMOKE"])) #Smoker or not
SCC = le.fit_transform(list(data["SCC"])) #Calories consumption monitoring
FAF = le.fit_transform(list(data["FAF"])) #Physical activity frequency
TUE = le.fit_transform(list(data["TUE"])) #Time using technology devices
CALC = le.fit_transform(list(data["CALC"])) #CALC
MTRANS = le.fit_transform(list(data["MTRANS"])) #Transportation used
Obesity = le.fit_transform(list(data["NObeyesdad"])) #Obesity level deducted

X = data.drop("NObeyesdad", axis=1)

predict = "NObeyesdad" # label we are tyring to predict

X = list(zip(Gender, Age, Height, Weight, family_history_with_overweight, 
              FCVC, NCP, CAEC, SMOKE, SCC, FAF,  CALC, MTRANS)) # convert all into ONE big list
y = list(Obesity)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

X = np.array(X)
y = np.array(y)


scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


log_regression = LogisticRegression(max_iter=800, C=95 , penalty='l2', solver='lbfgs')
log_regression.fit(x_train_scaled, y_train)
y_pred = log_regression.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 score:", f1)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# Assuming data["NObeyesdad"] contains strings like 'Normal_Weight', 'Overweight_Level_I', etc.
unique_classes = data["NObeyesdad"].unique()
sorted_classes = np.sort(unique_classes)  # Sort classes if necessary

# Now use 'sorted_classes' as your labels
labels = sorted_classes.tolist()  # Convert to list if needed

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=le.transform(sorted_classes))

# Plot confusion matrix
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Obesity Level')
plt.ylabel('True Obesity Level')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Calculate per-class accuracy if needed
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

# Plot per-class accuracy
plt.figure(figsize=(10, 6))
plt.bar(range(len(per_class_accuracy)), per_class_accuracy, color='skyblue', tick_label=labels)
plt.xlabel('Obesity Levels')
plt.ylabel('Accuracy')
plt.title('Per-class Accuracy for Obesity Levels')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


coefficients = log_regression.coef_
avg_importance = np.mean(np.abs(coefficients), axis=0)

feature_names = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 
                  'FCVC', 'NCP', 'CAEC', 'SMOKE', 'SCC', 'FAF',  'CALC', 'MTRANS']

# Create a DataFrame to store feature importance
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': avg_importance})

# Sort the DataFrame by importance in ascending order
feature_importance = feature_importance.sort_values('Importance', ascending=True)
print("Length of feature_names:", len(feature_names))
print("Length of avg_importance:", len(avg_importance))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()






