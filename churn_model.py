# importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# load dataset
df = pd.read_csv("C:\\Users\\Admin\\Customer-Churn-Analysis\\data\\churn.csv")
df.head()

# check basic info
df.info()
df.isnull().sum()

# handle missing values
# numerical -> fill with mean
df.fillna(df.mean(numeric_only = True), inplace = True)

# categorical -> fill with mode
for col in df.select_dtypes(include = 'object'):
  df[col] = df[col].fillna(df[col].mode()[0])

# encode categorical columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.select_dtypes(include = 'object'):
  df[col] = le.fit_transform(df[col])

# seperating features and target
X = df.drop("Churn", axis = 1)
y = df["Churn"]

# spitting train data and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42, stratify = y
)

# feature importance
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index = X.columns)
importances.sort_values(ascending = False).head(10)

# model training
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# model evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC: ", roc_auc_score(y_test, y_prob))