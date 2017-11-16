import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
fields = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=fields)
#Describes the dimensions of the dataset
print(dataset.shape)
#Get first 20 rows from the dataset
print(dataset.head(20))
# Get basic statistics about the dataset
print(dataset.describe())
# Class distribution
print(dataset.groupby('class').size())

# box and whisker plots
# Outliers are points that are atleast 3/2 times below min or 3/2 times above max.
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

# The sepel-length and sepal-width have a guassian distribution
dataset.hist()
plt._show()

# Multi-variate plot
# scatter plot matrix
scatter_matrix(dataset)
plt.show()
