'''
HR ANALYTICS OR PEOPLE ANALYTICS(A DATA DRIVEN APPROACH IN MANAGING PEOPLE AT WORK)

To know what type of problem we are dealing with or trying to predict
we can seperate into 2 categories:  Target = Churn or Turnover
                                    Features = Otherthan churn
Different problems approached are:
    Hiring/Assessment
    Retention
    Performance evaluation
Introduction:
    this is a project to predict the Employee churn(Employee attrition/ employee turnover) in advance, so that the 
    organisation can take necessary steps to reduce the higher cost in employing new candidates and putting efforts not 
    to loose efficient employess.

Objective:
    From the past data we need to consider the attributes relevant to work with and perform some data manipulation and data
    preprocessing in order to develope a robust model for the prediction of Employee churn. Which features are 
    affecting more for the employee attrition.
    
Structure of data:
    here we have Categorical and Mathematical variables.
    
print(data.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 14999 entries, 0 to 14998
Data columns (total 10 columns):
satisfaction            14999 non-null float64
evaluation              14999 non-null float64
number_of_projects      14999 non-null int64
average_montly_hours    14999 non-null int64
time_spend_company      14999 non-null int64
work_accident           14999 non-null int64
churn                   14999 non-null int64
promotion               14999 non-null int64
department              14999 non-null object
salary                  14999 non-null object
dtypes: float64(2), int64(6), object(2)

    This shows the variables of our dataset, last two coloumns show they are categorical variables(either ordinal-ranked 
    or Nominal-No intrinsic orfer), so we have to work with department and salary attributes to convert them into 
    numerical variables.
    
    print(data.department.unique()) ------ Nominal values
['sales' 'accounting' 'hr' 'technical' 'support' 'management' 'IT'
 'product_mng' 'marketing' 'RandD']

print(data.salary.unique()) ------ Ordinal values
['low' 'medium' 'high']

About the Predictive Model:
    Since it is a classification problem will the employee churn(leave the company or stay) classified primarity as 0 and
    1, it falls into binary classification. Since target is there it is a supervised classification.
    
    In Classification we have different types of models:
        a) Logistic Regression
        b) Dessision Tree
        c) Deep Learning
        d) Neural Networks
        e) Support Vector Machines
    We are moving ahead with Decision Tree Model: last leaf will be split to get the pure sample for the model.
    DT uses 2 splitting rules:
        1) Entropy:
                    -p*log(p) - (1-p)*log(1-p)
        
        2) Gini:
                    2*p*(1-p)
        
Decision Tree: 
    It can be used for the classification and regression problem.(by the CART model). Where the tree can
    be divided and subdivided in subsets of data by establising a proper linkage between the features 
    and the nodes to predict the final output as a value based from the given set of descrete values.
        
Overfitting:
    This is the most concerning problem when we work with decision trees. 
    Initial Score:
        classifier.score(feature_test, target_test)*100
        Out[71]: 97.97333333333333

        classifier.score(feature_train, target_train)*100
        Out[72]: 100.0
    How do we know if our model is overfitting:
        after training the test & train datasets and run the model if 
        Train shows Accuracy = 100%
        and when tested the same on Test Dataset Accuracy = 98 (or) 97 i.e comparitively less it means model is overfit.
        
    Solution:
        Limit the max_depth
        Limit the size of samples in leaves
    Later scores: 
        cross_val_score(model_depth_5, feature_test, target_test)*100
        Out[69]: array([97.76179057, 97.04      , 97.75820657])

        cross_val_score(model_sample_100, feature_test, target_test)*100
        Out[70]: array([95.36370903, 94.16      , 95.67654123])
        
        
Evaluating the Model:
    Prediction Errors:
        COnfusion Matrix:
            
            |____0____Actual____1____|
            |           |            |
         0  |     TN    |     FN     |
Predicted   |___________|____________|
            |           |            |
          1 |     FP    |     TP     |
            |___________|____________|
            
      Metrics 1:      
        If the target is leavers:(focus on FN)
            Recall Score = TP/(TP+FN)
            
        If the target is leavers:(focus on FP)
            Specificity = TN/(TN+FP)
        
     Metrics 2:       
        If the target is leavers: (focus on FP)
            Precision Score = TP/(TP+FP)
            
Imbalance in the Model:
    In the model the stayers are having higher percentage than the leavers. So the problem of imbalance is because of
    the higher weightage put by the stayers is 75%(so it predicts the stayers correctly) and leavers weightage is 25%
    (shows the probability of predicting leavers is not always correct)
    
    to correct this while building the model just balance the class weight.
    
Hyperparameter Tuning:
    previously by using the train/test split ensures that the training data is not overfitting
    it consists in tuning the model to get the best prediction results on the test set. Therefore, it is recommended 
    to validate the model on different testing sets. K-fold cross-validation allows us to achieve this:

    it splits the dataset into a training set and a testing set
    it fits the model, makes predictions and calculates a score (you can specify if you want the accuracy, precision, 
    recall...)
    it repeats the process k times in total
    it outputs the average of the 10 scores
            
'''


import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sb
from sklearn.model_selection import cross_val_score #Accuracy classification of the model
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn import tree

f_read = pd.read_csv("churn_year.csv")

#To look at the types of attributes
print(f_read.info())
f_read.head()

#look for the types of departments and salary
print(f_read.department.unique())

print(f_read.salary.unique())

#Now transform the categorical variables into numerical variables
f_read.salary = f_read.salary.astype('category')

f_read.salary = f_read.salary.cat.reorder_categories(['low', 'medium', 'high'])
          

#by using cat.reorder_categories(method) we can assign the order in numerical format
f_read.salary = f_read.salary.cat.codes
print(f_read.salary)

#for Nominal category variable you should creae dummies and store in another dataframe
nom_dept = pd.get_dummies(f_read.department)
print(nom_dept)

#here we observe the dummy trap in the column where it shows 1 (here its 'sales' column in department_all dataframe.)
'''

department_all.head()
Out[12]: 
   IT  RandD  accounting  hr  management  marketing  product_mng  sales  \
0   0      0           0   0           0          0            0      1   
1   0      0           0   0           0          0            0      1   
2   0      0           0   0           0          0            0      1   
3   0      0           0   0           0          0            0      1   
4   0      0           0   0           0          0            0      1   

   support  technical  
0        0          0  

'''

nom_dept = nom_dept.drop("sales", axis = 1)
nom_dept.head()

'''
department_all.head()
Out[14]: 
   IT  RandD  accounting  hr  management  marketing  product_mng  support  \
0   0      0           0   0           0          0            0        0   
1   0      0           0   0           0          0            0        0   
2   0      0           0   0           0          0            0        0   
3   0      0           0   0           0          0            0        0   
4   0      0           0   0           0          0            0        0   

   technical  
0          0  

'''

#now drop the department column from the data table and insert department_all data into data table

f_read = f_read.drop("department", axis = 1)
print(f_read.info())


#joining 2 tables
f_read = f_read.join(nom_dept)
f_read.tail()

'''
Now lets deep dive into data we need to predict the churn of employees. following steps calculate the percentage of
churn for the given datapoints.

'''

cal_churn_val = (f_read.churn.value_counts())/len(f_read)*100

print(cal_churn_val)

#Using the visualization libraries in pandas 
correlation_matrix = f_read.corr()
sb.heatmap(correlation_matrix)
plot.show()


#developing a model: Preparing the variables and splitting data into test and train datasets

print(f_read.head())
y = f_read.churn
X = f_read.drop("churn", axis = 1)

from sklearn.model_selection import train_test_split
y_train, y_test, X_train, X_test = train_test_split(y, X, train_size = 0.80)

#Implementation of Decision Tree Model 
from sklearn.tree import DecisionTreeClassifier

#random state is said so that if we run it next time also we will get the same results
classifier = DecisionTreeClassifier(random_state = 50) 

#we are making the train dataset learn the model
classifier.fit(X_train, y_train)

#we are testing the accuracy with the test data.
classifier.score(X_train, y_train)*100
classifier.score(X_test, y_test)*100

# Import the graphical visualization export function
from sklearn.tree import export_graphviz

# Export the tree to a dot file
export_graphviz(classifier,"tree.dot")

#Tuning employee churn classifier this is the solution to overfitting of the model
#Pruning the Model by selecting the best number for max_depth:
model_depth_5 = DecisionTreeClassifier(max_depth = 5, random_state = 42)

#To avoid overfitting here we are defining the min samples per leaf to 100, this is other way
model_sample_100 = DecisionTreeClassifier(min_samples_leaf = 100, random_state = 42)

#Here the results show that the variance between the test and train is relative less. Which means overfitting is avoided.
cross_val_score(model_depth_5, X_test, y_test)*100

cross_val_score(model_sample_100, X_test, y_test)*100

# Setting all parameters develope a model
mod_1 = DecisionTreeClassifier(max_depth=7, class_weight="balanced", random_state=42)
# fit the model on the train dataset 
mod_1.fit(X_train,y_train)
# Predict on the test set component, which further used for cal. score
prediction_b = mod_1.predict(X_test)
# Print the recall score for the predicted model
recall_score(y_test,prediction_b)
# Print the ROC/AUC score for the predicted model
roc_auc_score(y_test, prediction_b)

#Reducing the problem of Hyperparameter tuning is by introducing the k-fold cross vallidation.
from sklearn.model_selection import cross_val_score
cross_val_score(classifier,X,y,cv=15)

# import the GridSearchCV function
from sklearn.model_selection import GridSearchCV

# Get values for maximum depth, as we change the range accuracy of the model varies
#It varies fro model to model based on data and prediction we are choosing
depth = [i for i in range(5,30,3)]

# Gives the values for minimum sample size, as we change the range accuracy of the model varies
samples = [i for i in range(5,500,50)]

# Create the dictionary with parameters to be checked
parameters = dict(max_depth=depth, min_samples_leaf=samples)

# set up parameters: done
parameters = dict(max_depth=depth, min_samples_leaf=samples)
  
# initialize the param object using the GridSearchCV function, initial model and parameters above
param = GridSearchCV(classifier, parameters)
print(param)

# fit the param_search to the training dataset
print(param.fit(X_train, y_train))


# Calculate feature importances
X_importances = classifier.feature_importances_

# Create a list of features
X_list = list(X)

# Save the results inside a DataFrame using feature_list as an indnex
DT_importance = pd.DataFrame(index=X_list, data=X_importances, columns=["importance"])

# Sort values to learn most important features
DT_importance.sort_values(by="importance", ascending=False)

# select only features with importance higher than 5%
final_X = DT_importance[DT_importance.importance>0.05]
print(final_X)
# final list of features is created
final_list = final_X.index

# transform both features_train and features_test components to include only selected features
features_train_selected = X_train[final_list]
features_test_selected = X_test[final_list]


# As per the best parameters given above model has been initialized
model_best = DecisionTreeClassifier(max_depth=10, min_samples_leaf=150, class_weight="balanced", random_state=50)

# Fit the model using only selected features from training set: done
model_best.fit(features_train_selected, y_train)

# Make prediction based on selected list of features from test set
prediction_best = model_best.predict(features_test_selected)

# Print the general accuracy of the model_best
print(model_best.score(features_test_selected, y_test) * 100)

'''
#Final visualization
visual_graph = tree.export_graphviz(model_best, out_file='tree.dot', filled=True, rounded=True,
                                feature_names=final_list,  
                                class_names="churn")
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import pydot
dot_data = export_graphviz(model_best, out_file=None,
                filled=True, rounded=True,
                special_characters=True)
graph = pydot.graph_from_dot_data(dot_data)
dot_data(graph.create_png())

'''




























