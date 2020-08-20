import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go

data = pd.read_csv('online_shoppers_intention.csv')
#Shape of data

#Exploitary analysis


print('The shape of the dataset : ', data.shape)

data.head()

#View summary of dataset
data.info()

#Check the data types of columns
data.dtypes

#View statistical properties of dataset
data.describe()

#checking for missing value
data.isnull().sum()

#checking categorical values
categorical = [var for var in data.columns if data[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)

#Preview categorical variables
data[categorical].head()

#Frequency distribution of categorical variables

for var in categorical: 
    
    print(data[var].value_counts())
    







# plotting a pie chart for visitors

plt.rcParams['figure.figsize'] = (18, 7)
size = [10551, 1694, 85]
colors = ['violet', 'magenta', 'pink']
labels = "Returning Visitor", "New_Visitor", "Others"
explode = [0, 0, 0.1]
plt.subplot(1, 2, 1)
plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')
plt.title('Different Visitors', fontsize = 30)
plt.axis('off')
plt.legend()


# plotting a pie chart for browsers
size = [7961, 2462, 736, 467,174, 163, 300]
colors = ['orange', 'yellow', 'pink', 'crimson', 'lightgreen', 'cyan', 'blue']
labels = "2", "1","4","5","6","10","others"

plt.subplot(1, 2, 2)
plt.pie(size, colors = colors, labels = labels, shadow = True, autopct = '%.2f%%', startangle = 90)
plt.title('Different Browsers', fontsize = 30)
plt.axis('off')
plt.legend()
plt.show()


# plotting a pie chart for different number of OSes users have.

size = [6601, 2585, 2555, 478, 111]
colors = ['orange', 'yellow', 'pink', 'crimson', 'lightgreen']
labels = "2", "1","3","4","others"
explode = [0, 0, 0, 0, 0]

circle = plt.Circle((0, 0), 0.6, color = 'white')

plt.subplot(1, 2, 1)
plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')
plt.title('Operating System Used by Visitors', fontsize = 30)
p = plt.gcf()
p.gca().add_artist(circle)
plt.axis('off')
plt.legend()

# plotting a pie chart for share of special days

size = [3364, 2998, 1907, 1727, 549, 448, 433, 432, 288, 184]
colors = ['orange', 'yellow', 'pink', 'crimson', 'lightgreen', 'cyan', 'magenta', 'lightblue', 'lightgreen', 'violet']
labels = "May", "November", "March", "December", "October", "September", "August", "July", "June", "February"
explode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

circle = plt.Circle((0, 0), 0.6, color = 'white')

plt.subplot(1, 2, 2)
plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')
plt.title('Special Days in Each Month', fontsize = 30)
p = plt.gcf()
p.gca().add_artist(circle)
plt.axis('off')
plt.legend()

plt.show()






# Informational duration vs revenue

plt.rcParams['figure.figsize'] = (18, 15)

plt.subplot(2, 2, 1)
sns.stripplot(data['Revenue'], data['Informational_Duration'], palette = 'rainbow')
plt.title('Info. duration vs Revenue', fontsize = 30)
plt.ylabel('Info. duration', fontsize = 15)
plt.xlabel('Revenue', fontsize = 15)


# Admonistration duration vs revenue

plt.subplot(2, 2, 2)
sns.stripplot(data['Revenue'], data['Administrative_Duration'], palette = 'pastel')
plt.title('Admn. duration vs Revenue', fontsize = 30)
plt.ylabel('Admn. duration', fontsize = 15)
plt.xlabel('Revenue', fontsize = 15)


# product related duration vs revenue

plt.subplot(2, 2, 3)
sns.stripplot(data['Revenue'], data['ProductRelated_Duration'], palette = 'dark')
plt.title('Product Related duration vs Revenue', fontsize = 30)
plt.ylabel('Product Related duration', fontsize = 15)
plt.xlabel('Revenue', fontsize = 15)

# exit rate vs revenue

plt.subplot(2, 2, 4)
sns.stripplot(data['Revenue'], data['ExitRates'], palette = 'spring')
plt.title('ExitRates vs Revenue', fontsize = 30)
plt.ylabel('ExitRates', fontsize = 15)
plt.xlabel('Revenue', fontsize = 15)

plt.show()








plt.rcParams['figure.figsize'] = (18, 7)

plt.subplot(1, 2, 1)
data["Informational"].value_counts().plot.bar(color="purple",figsize=(10,5))
plt.title("Informational plot")
plt.ylabel("Number of counts or visits")

plt.subplot(1, 2, 2)
sns.distplot(data["Informational_Duration"],kde=True,norm_hist=True)
plt.title("Time spent by the user on the website")
plt.grid()

plt.show()







plt.rcParams['figure.figsize'] = (18, 7)

plt.subplot(1, 2, 1)
sns.countplot(data['Revenue'], palette = 'pastel')
plt.title('Buy or Not', fontsize = 30)
plt.xlabel('Revenue or not', fontsize = 15)
plt.ylabel('count', fontsize = 15)

plt.subplot(1, 2, 2)
sns.countplot(data['Weekend'], palette = 'inferno')
plt.title('Purchase on Weekends', fontsize = 30)
plt.xlabel('Weekend or not', fontsize = 15)
plt.ylabel('count', fontsize = 15)

print(data['VisitorType'].value_counts())

plt.show()



# visualizing the distribution of customers around the Region

plt.rcParams['figure.figsize'] = (18, 7)

plt.subplot(1, 2, 1)
plt.hist(data['TrafficType'], color = 'lightgreen')
plt.title('Distribution of diff Traffic',fontsize = 30)
plt.xlabel('TrafficType Codes', fontsize = 15)
plt.ylabel('Count', fontsize = 15)

# visualizing the distribution of customers around the Region

plt.subplot(1, 2, 2)
plt.hist(data['Region'], color = 'lightblue')
plt.title('Distribution of Customers',fontsize = 30)
plt.xlabel('Region Codes', fontsize = 15)
plt.ylabel('Count', fontsize = 15)

plt.show()



# visitor type vs revenue

df = pd.crosstab(data['VisitorType'], data['Revenue'])
df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['lightgreen', 'green'])
plt.title('Visitor Type vs Revenue', fontsize = 30)
plt.show()




# weekend vs Revenue

df = pd.crosstab(data['Weekend'], data['Revenue'])
df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['orange', 'crimson'])
plt.title('Weekend vs Revenue', fontsize = 30)
plt.show()




# specials vs Revenue

df = pd.crosstab(data['SpecialDay'], data['Revenue'])
df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['red', 'blue'])
plt.title('Special days vs Revenue', fontsize = 30)
plt.show()


# Traffic Type vs Revenue

df = pd.crosstab(data['TrafficType'], data['Revenue'])
df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['lightpink', 'yellow'])
plt.title('Traffic Type vs Revenue', fontsize = 30)
plt.show()





# visitor type vs exit rates w.r.t revenue

plt.rcParams['figure.figsize'] = (18, 15)
plt.subplot(2, 2, 1)
sns.violinplot(x = data['VisitorType'], y = data['ExitRates'], hue = data['Revenue'], palette = 'rainbow')
plt.title('Visitors vs ExitRates wrt Rev.', fontsize = 30)

# visitor type vs exit rates w.r.t revenue
plt.subplot(2, 2, 2)
sns.violinplot(x = data['VisitorType'], y = data['PageValues'], hue = data['Revenue'], palette = 'gnuplot')
plt.title('Visitors vs PageValues wrt Rev.', fontsize = 30)

# region vs pagevalues w.r.t. revenue
plt.subplot(2, 2, 3)
sns.violinplot(x = data['Region'], y = data['PageValues'], hue = data['Revenue'], palette = 'Greens')
plt.title('Region vs PageValues wrt Rev.', fontsize = 30)

#region vs exit rates w.r.t. revenue
plt.subplot(2, 2, 4)
sns.violinplot(x = data['Region'], y = data['ExitRates'], hue = data['Revenue'], palette = 'spring')
plt.title('Region vs Exit Rates w.r.t. Revenue', fontsize = 30)

plt.show()


# lm plot

plt.rcParams['figure.figsize'] = (20, 10)

sns.lmplot(x = 'Administrative', y = 'Informational', data = data, x_jitter = 0.05)
plt.title('LM Plot between Admistrative and Information', fontsize = 15)

plt.show()




# page values vs revenue

plt.rcParams['figure.figsize'] = (18, 7)

plt.subplot(1, 2, 1)
sns.stripplot(data['Revenue'], data['PageValues'], palette = 'autumn')
plt.title('PageValues vs Revenue', fontsize = 30)
plt.ylabel('PageValues', fontsize = 15)
plt.xlabel('Revenue', fontsize = 15)

# bounce rates vs revenue
plt.subplot(1, 2, 2)
sns.stripplot(data['Revenue'], data['BounceRates'], palette = 'magma')
plt.title('Bounce Rates vs Revenue', fontsize = 30)
plt.ylabel('Boune Rates', fontsize = 15)
plt.xlabel('Revenue', fontsize = 15)

plt.show()






# region vs Revenue

df = pd.crosstab(data['Region'], data['Revenue'])
df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['lightblue', 'blue'])
plt.title('Region vs Revenue', fontsize = 30)
plt.show()



# month vs pagevalues wrt revenue

plt.rcParams['figure.figsize'] = (18, 15)
plt.subplot(2, 2, 1)
sns.boxplot(x = data['Month'], y = data['PageValues'], hue = data['Revenue'], palette = 'inferno')
plt.title('Mon. vs PageValues w.r.t. Rev.', fontsize = 30)

# month vs exitrates wrt revenue
plt.subplot(2, 2, 2)
sns.boxplot(x = data['Month'], y = data['ExitRates'], hue = data['Revenue'], palette = 'Reds')
plt.title('Mon. vs ExitRates w.r.t. Rev.', fontsize = 30)

# month vs bouncerates wrt revenue
plt.subplot(2, 2, 3)
sns.boxplot(x = data['Month'], y = data['BounceRates'], hue = data['Revenue'], palette = 'Oranges')
plt.title('Mon. vs BounceRates w.r.t. Rev.', fontsize = 30)

# visitor type vs exit rates w.r.t revenue
plt.subplot(2, 2, 4)
sns.boxplot(x = data['VisitorType'], y = data['BounceRates'], hue = data['Revenue'], palette = 'Purples')
plt.title('Visitors vs BounceRates w.r.t. Rev.', fontsize = 30)

plt.show()


# Inputing Missing Values with 0

data.fillna(0, inplace = True)
print(data.isnull().sum().sum())

# Q1: Time Spent by The Users on Website vs Bounce Rates


# let's cluster Administrative duration and Bounce Ratw to different types of clusters in the dataset.

x = data.iloc[:, [1, 6]].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i,
              init = 'k-means++',
              max_iter = 300,
              n_init = 10,
              random_state = 0,
              algorithm = 'elkan',
              tol = 0.001)
    km.fit(x)
    labels = km.labels_
    wcss.append(km.inertia_)
    
plt.rcParams['figure.figsize'] = (15, 7)
plt.plot(range(1, 11), wcss)
plt.grid()
plt.tight_layout()
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()

#visualizing the cluster using scatter plot

km = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'Un-interested Customers')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'General Customers')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'cyan', label = 'Target Customers')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')

plt.title('Administrative Duration vs Bounce Rate', fontsize = 20)
plt.grid()
plt.xlabel('Administrative Duration')
plt.ylabel('Bounce Rates')
plt.legend()
plt.show()

# informational duration vs Bounce Rates
x = data.iloc[:, [3, 6]].values

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i,
              init = 'k-means++',
              max_iter = 300,
              n_init = 10,
              random_state = 0,
              algorithm = 'elkan',
              tol = 0.001)
    km.fit(x)
    labels = km.labels_
    wcss.append(km.inertia_)
    
plt.rcParams['figure.figsize'] = (15, 7)
plt.plot(range(1, 11), wcss)
plt.grid()
plt.tight_layout()
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()

#visualizing the cluster using scatter plot

km = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'Un-interested Customers')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'Target Customers')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')

plt.title('Informational Duration vs Bounce Rates', fontsize = 20)
plt.grid()
plt.xlabel('Informational Duration')
plt.ylabel('Bounce Rates')
plt.legend()
plt.show()

# productrelated duration vs Bounce Rates
x = data.iloc[:, [5, 6]].values

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i,
              init = 'k-means++',
              max_iter = 300,
              n_init = 10,
              random_state = 0,
              algorithm = 'elkan',
              tol = 0.001)
    km.fit(x)
    labels = km.labels_
    wcss.append(km.inertia_)
    
plt.rcParams['figure.figsize'] = (15, 7)
plt.plot(range(1, 11), wcss)
plt.grid()
plt.tight_layout()
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()

#visualizing the cluster using scatter plot

km = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'Un-interested Customers')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'Target Customers')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')

plt.title('Productrelated Duration vs Bounce Rates', fontsize = 20)
plt.grid()
plt.xlabel('Product related Duration')
plt.ylabel('Bounce Rates')
plt.legend()
plt.show()



from scipy import stats
def out_lier(df,colname):
    qua1 = df[colname].quantile(0.25)
    qua2 = df[colname].quantile(0.75)
    t=3

    z = np.abs(stats.zscore(df[colname]))
    df[colname][z>t] = qua2
    df[colname][z<-t] = qua1

#out_lier(data1,'Administrative')
out_lier(data,'Administrative_Duration')
#out_lier(data1,'Informational')
out_lier(data,'Informational_Duration')
#out_lier(data1,'ProductRelated')
out_lier(data,'ProductRelated_Duration')


#dummy variables
categorical = ['VisitorType','Month']
data = pd.get_dummies(data,columns = categorical,drop_first=True)

#data = preprocessing.StandardScaler().fit(data).transform(data.astype(float))
print(data)
print(data.shape)

# label encoding of revenue

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Revenue'] = le.fit_transform(data['Revenue'])

# removing the target column revenue from x
x = data.drop('Revenue', axis = 1)

y = data['Revenue']


# checking the shapes
print("Shape of x:", x.shape)
print("Shape of y:", y.shape)

# splitting the data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 2)

print("Shape of x_train :", x_train.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_test :", y_test.shape)


# MODELLING RANDOMFOREST

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# evaluating the model
rantrain=model.score(x_train, y_train)
print("Training Accuracy :", rantrain)
rantest=model.score(x_test, y_test)
print("Testing Accuracy :", rantest)


# confusion matrix
print(confusion_matrix(y_test,y_pred))

# classification report
cr = classification_report(y_test, y_pred)
print(cr)

# finding the Permutation importance

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, random_state = 0).fit(x_test, y_test)
eli5.show_weights(perm, feature_names = x_test.columns.tolist())



# importing pdp
from pdpbox import pdp, info_plots

base_features = x_test.columns.values.tolist()

feat_name = 'Administrative_Duration'
pdp_dist = pdp.pdp_isolate(model=model, dataset=x_test, model_features = base_features, feature = feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()


# plotting partial dependency plot for Informational Duration

base_features = x_test.columns.tolist()

feat_name = 'Informational_Duration'
pdp_dist = pdp.pdp_isolate(model, x_test, base_features, feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()


# plotting partial dependency plot for Product related Duration

base_features = x_test.columns.tolist()

feat_name = 'ProductRelated_Duration'
pdp_dist = pdp.pdp_isolate(model, x_test, base_features, feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()


# let's take a look at the shap values

# importing shap
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_test)

shap.summary_plot(shap_values[1], x_test, plot_type = 'bar')


# let's create a function to check the customer's conditions


def customer_analysis(model, customer):
  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(customer)
  shap.initjs()
  return shap.force_plot(explainer.expected_value[1], shap_values[1], customer)

customers = x_test.iloc[1,:].astype(float)
customer_analysis(model, customers)

customers = x_test.iloc[1531,:].astype(float)
customer_analysis(model, customers)


shap_values = explainer.shap_values(x_train.iloc[:100])
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], x_test.iloc[:100])




#KNN ALGORITHM

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
Y_pred=model.predict(x_test)
knntrain=model.score(x_train, y_train)
print("Training Accuracy :", knntrain)
knntest=model.score(x_test, y_test)
print("Testing Accuracy :", knntest)


#SUPPORT VECTOT MACHINE

from sklearn import svm
model = svm.SVC(kernel='rbf')
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
svmtrain=model.score(x_train, y_train)
print("Training Accuracy :", svmtrain)
svmtest=model.score(x_test, y_test)
print("Testing Accuracy :", svmtest)


#NAIVE BAYES ALGORITHM

from sklearn.naive_bayes import GaussianNB
model =GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
nbtrain=model.score(x_train, y_train)
print("Training Accuracy :", nbtrain)
nbtest=model.score(x_test, y_test)
print("Testing Accuracy :", nbtest)

#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
lrtrain=model.score(x_train, y_train)
print("Training Accuracy :", lrtrain)
lrtest=model.score(x_test, y_test)
print("Testing Accuracy :", lrtest)

#comparisson between algorithm

#train plot
objects=['RANDOMFOREST','KNN','SVM','NAIVE','LOGISTIC']
y_pos=np.arange(len(objects))
trainaccuracy=[rantrain,knntrain,svmtrain,nbtrain,lrtrain]

plt.bar(y_pos, trainaccuracy, align='center', alpha=0.5,color=('red','blue','green','orange','purple'))
plt.xticks(y_pos, objects)
plt.xlabel('ALGORITHM')
plt.ylabel('TRAIN ACCURACY')
plt.title('Comparisson Between Algorithms')
plt.show()

#test plot
objects1=['RANDOMFOREST','KNN','SVM','NAIVE','LOGISTIC']
y_pos1=np.arange(len(objects1))
testaccuracy=[rantest,knntest,svmtest,nbtest,lrtest]

plt.bar(y_pos1, testaccuracy, align='center', alpha=0.5,color=('red','blue','green','orange','purple'))
plt.xticks(y_pos, objects)
plt.xlabel('ALGORITHM')
plt.ylabel('TEST ACCURACY')
plt.title('Comparisson Between Algorithms')
plt.show()


