# anamolydetection
AnomaData: Predictive Maintenance Solution utilizing machine learning for automated anomaly detection in equipment, focusing on data exploration, preprocessing, and logistic regression modeling.
#Import The Necessary Libraries
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;
import scipy.stats as scp;
from sklearn.feature_selection import RFE;
from sklearn.preprocessing import StandardScaler;
from sklearn.preprocessing import OneHotEncoder;
from sklearn.compose import ColumnTransformer;
from sklearn.preprocessing import LabelEncoder;
from sklearn.linear_model import LinearRegression;
from sklearn.model_selection import train_test_split;
from sklearn.ensemble import ExtraTreesRegressor;
from sklearn.metrics import r2_score,accuracy_score,precision_score,recall_score,f1_score, roc_auc_score;
import statsmodels.api as apl;
from sklearn.preprocessing import StandardScaler;
from sklearn.linear_model import LogisticRegression;
from sklearn.model_selection import cross_val_score,KFold;
from sklearn.model_selection import StratifiedKFold;
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,classification_report;
from imblearn.over_sampling import SMOTE ;
from imblearn.under_sampling import NearMiss;
import collections;
from sklearn.linear_model import Ridge;
from numpy import absolute;
from sklearn.model_selection import GridSearchCV;
from numpy import arange;
from sklearn.preprocessing import PowerTransformer;
from sklearn.linear_model import Lasso;
from sklearn.linear_model import ElasticNet;
from sklearn.preprocessing import PowerTransformer;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.svm import SVC;
from sklearn.ensemble import IsolationForest;
from sklearn.preprocessing import FunctionTransformer;
from sklearn.decomposition import PCA;
from sklearn.model_selection import cross_val_score;
import pickle;
import warnings;

warnings.filterwarnings('ignore')




# Performing Data Preprocessing

df = pd.read_csv("AnomaData.csv");
print(df.info())
print(df.shape)
print(df.describe())
print(df)
print(df.dropna(inplace = True))
print(df.drop_duplicates())
print("Duplicates in the Data",df[df.duplicated()])
df['time'] = pd.to_datetime(df['time'])
print(df['time'].dtype)

# Extracting Numerical , Discrete, Categorical Features
feature_having_na = [feature for feature in df.columns if df[feature].isnull().sum()>1]
print('feature having na:- ', feature_having_na);
numerical_feature = [feature for feature in df.columns if df[feature].dtypes !='O' ]
print('numerical features are ', numerical_feature, 'count of numerical featuers are ', len(numerical_feature))
categorical_feature = list(set(df.columns)-set(numerical_feature))
print('categorical_feature are ', categorical_feature, 'count of categorical_feature are ', len(categorical_feature))

descrete_features = [feature for feature in numerical_feature if len(df[feature].unique())<25]
print('descrete features are ', descrete_features, 'count of descrete_features are ', len(descrete_features))
# Handling missing values
# Drop rows with missing values

#finding the null values in the data.
print(df.isnull().sum())
df= df.drop(["time"],axis=1)
print("skewness in the data ", df.skew())

# Analysing the skewness
for c in df.select_dtypes(include=[np.number]).columns:
  if c not in descrete_features:

    sns.histplot(df[c])
    plt.title(c)
    plt.show()

# Reducing the Skewness of the features
for j in df.select_dtypes(include=[np.number]).columns:
    if j not in descrete_features:

      unique_values = df[j].nunique()

      if (unique_values >= 20):
        if df[j].skew()<(-0.5) :
            print("before skew",df[j].skew())
            df[j],lam = scp.yeojohnson(df[j])
            print("skewed value ",df[j].skew())
            print(df[j].nunique())

        elif df[j].skew()>0.5:
            print(" before skew",df[j].skew())
            df[j],lam1 = scp.yeojohnson(df[j])
            print("skewed value ",df[j].skew())
            print(df[j].nunique())


        else :
          pass;

  # Classifying dependent and independent variables
  # by analysing I found that the outliers in the data are useful for getting the good accuracy so removing the outliers is not necessary oin this model .

x=df.drop(["y"],axis=1)
y=df["y"]

print(x)
print(len(y))

# Counting the no of labels in the dependent variable
counter = collections.Counter(y)
print(" value counts",counter)

# The data in the dependent feature is not balanced so balancing the data using the SMOTE overfitting technique.
sm = SMOTE(random_state = 2)
x_res, y_res = sm.fit_resample(x, y.ravel())
print('After OverSampling, the shape of X: {}'.format(x_res.shape))
print('After OverSampling, the shape of y: {} \n'.format(y_res.shape))
print("After OverSampling, counts of label '1': {}".format(sum(y_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_res == 0)))
counter = collections.Counter(y_res)
print(" value counts",counter)
# Performing PCA  
pca = PCA(n_components=12)

# Fit PCA on the data
pca.fit(x_res)

# Transform the data onto the new feature space
X_pcg = pca.transform(x_res)

# Output the explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Display the transformed data
print("Transformed data shape:", X_pcg.shape)
print("Transformed data:")
print(X_pcg)
# Splitting the data into training and testing
x_train1,x_test1,y_train1,y_test1=train_test_split(X_pcg,y_res,test_size=0.2)
print(x_train1)


# Building and Training the Model

log_res = LogisticRegression()
log_res.fit(x_train1,y_train1)

log_pres= log_res.predict(x_test1)

print(" Test accuracy is ", accuracy_score(y_test1,log_pres))
print(" Train accuracy is ", accuracy_score(y_train1 ,log_res.predict(x_train1)))
print(confusion_matrix(y_test1,log_pres))

clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(x_train1,y_train1)

# Make predictions on the testing set
y_pred = clf.predict(x_test1)

# Evaluate the model
accuracy = accuracy_score(y_test1, y_pred)
conf_matrix = confusion_matrix(y_test1, y_pred)
classification_rep = classification_report(y_test1, y_pred)

# Print the results
print("Decision tree model")
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

rfs = RandomForestClassifier()
rfs.fit(x_train1,y_train1)
rfs_pre = rfs.predict(x_test1)
print("random forest classifier ")

accuracy_rfs = accuracy_score(y_test1,rfs_pre)
conf_matrix_rfs = confusion_matrix(y_test1,rfs_pre)
classification_rep_rfs = classification_report(y_test1,rfs_pre)

# Print the results
print(f"Accuracy: {accuracy_rfs}")
print("Confusion Matrix:")
print(conf_matrix_rfs)
print("Classification Report:")
print(classification_rep_rfs)

# We have used PCA for feature extraction and the model performed well with Feature Extraction Technique . lets build the model using Feature Selection Technique.
x_train,x_test,y_train,y_test=train_test_split(x_res,y_res,test_size=0.2)
print(x_train)
# standardising the data
sc= StandardScaler()
x_train=sc.fit_transform(x_train)
x_test= sc.transform(x_test)
print(x_train)

# Getting the important features using Recursive Feature Elimination Method 
log = LogisticRegression()
rfe=RFE(log)
l=rfe.fit(x,y)

print("Num Features: %d" % l.n_features_)
print("Selected Features",x.columns[l.support_])
print("Feature Ranking: %s" % l.ranking_)

# Getting the Feature Importance using ExtraTree Classifier 
c= ExtraTreesRegressor()
lc=c.fit(x,y)
print(x.columns)
print(lc.feature_importances_)



log_reg = LogisticRegression(penalty='l2')


log_reg.fit(x_train,y_train)
log_pre= log_reg.predict(x_test)

print(" Test accuracy is ", accuracy_score(y_test,log_pre))
print(" Train accuracy is ", accuracy_score(y_train ,log_reg.predict(x_train)))
print(confusion_matrix(y_test,log_pre))
# Evaluate the model
accuracy = accuracy_score(y_test,log_pre)
precision = precision_score(y_test,log_pre)
recall = recall_score(y_test, log_pre)
f1 = f1_score(y_test, log_pre)
roc_auc = roc_auc_score(y_test,log_pre)
print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
# Hyper Parameter Tuning

k_fold= 10


kf = KFold(n_splits= k_fold, shuffle = True, random_state= 42 )


cross_val_res = cross_val_score(log_reg, x, y , cv= kf)



print("cross validation results " ,cross_val_res)
print("mean accuracy ",cross_val_res.mean() )
print("f1 score",f1_score(y_test, log_pre,average='macro'))
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_dist = {
    "max_depth": [3, None],
    "max_features": randint(1, 9),
    "min_samples_leaf": randint(1, 9),
    "criterion": ["gini", "entropy"]
}

tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
tree_cv.fit(x_train, y_train)

print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
clf1 = DecisionTreeClassifier(criterion= 'gini',random_state=42)

# Train the classifier
clf1.fit(x_train,y_train)

# Make predictions on the testing set
y_pred = clf1.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print("Decision tree model")
print(f"Accuracy for test: {accuracy}")
print("Accuracy for train", accuracy_score(y_train,clf1.predict(x_train)) )
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_model = RandomForestClassifier()

tree_cv = RandomizedSearchCV(rf_model, param_grid, cv=5)
tree_cv.fit(x_train, y_train)

print("Tuned Random Forest  Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))



