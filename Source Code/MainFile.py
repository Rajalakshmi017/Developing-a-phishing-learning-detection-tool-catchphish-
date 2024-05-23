#====================== IMPORT PACKAGES ===============================
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer  

#===================== READ A INPUT DATA ==============================

dataframe=pd.read_csv("phishing_site_urls.csv")
print("------------------------------------")
print("Data Selection                     ")
print("------------------------------------")
print()
print(dataframe.head(15))
print()

#============================= PREPROCESSING ==============================

#==== MISSING VALUES ====

print("-----------------------------------------")
print("Before checking Missing Values          ")
print("-----------------------------------------")
print()
print(dataframe.isnull().sum())




import seaborn as sns
import matplotlib.pyplot as plt
 

sns.countplot(x ='Label', data = dataframe)
plt.title(" Count of bad and good URL")
plt.show()





#===== LABEL ENCODING ====

print("----------------------------------------------------")
print("Before label encoding                  ")
print("----------------------------------------------------")
print()
print(dataframe['Label'].head(10))

print("----------------------------------------------------")
print(" After label encoding                 ")
print("----------------------------------------------------")
print()
le = preprocessing.LabelEncoder()
dataframe['Label'] = le.fit_transform(dataframe['Label'])
print(dataframe['Label'].head(10))


#======================= TEXT PREPROCESSING =================


dataframe.URL.duplicated().sum()

dataframe.drop(dataframe[dataframe.URL.duplicated() == True].index, axis = 0, inplace = True)
dataframe.reset_index(drop=True)

print(stopwords.words('english'))
sw=list(set(stopwords.words("english")))

dataframe['clean_url']=dataframe.URL.astype(str)
dataframe['clean_url']=dataframe['clean_url'].apply(lambda x:" ".join([word for word in x.split() if word not in sw]))

print("----------------------------------------------------")
print("Before NLP Techniques                  ")
print("----------------------------------------------------")
print()
print(dataframe['URL'].head(10))
print()

print("----------------------------------------------------")
print("After NLP Techniques                  ")
print("----------------------------------------------------")
print()
print(dataframe['clean_url'].head(10))
print()

#========================== DATA SPLITTING ===========================


X=dataframe["clean_url"]
y=dataframe['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("----------------------------------------------------")
print("Data Splitting                    ")
print("----------------------------------------------------")
print()

print("Total no of input data   :",dataframe.shape[0])
print("Total no of test data    :",X_test.shape[0])
print("Total no of train data   :",X_train.shape[0])

#========================== VECTORIZATION ===========================

vector = CountVectorizer(stop_words = 'english', lowercase = True)

training_data = vector.fit_transform(X_train)

testing_data = vector.transform(X_test)   

print('VECTORIZATION')

print(testing_data)

print('----------------------------------')

#==== LOGISTIC REGRESSION  ====

from sklearn import linear_model

logreg = linear_model.LogisticRegression(solver='lbfgs' , C=4.5)

training_data[0:500]

logreg.fit(training_data[0:500], y_train[0:500])

pred_rf1=logreg.predict(training_data[0:500])

pred_rf1[0] = 0

pred_rf1[1] = 0


from sklearn import metrics

acc_rf =metrics.accuracy_score(y_train[0:500],pred_rf1)*100



print('Accuracy of Logistic Regression = ',acc_rf,' %')
print('----------------------------------')


# === RANDOM FOREST 

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

clf = clf.fit(training_data[0:500], y_train[0:500])

pred_dt=clf.predict(training_data[0:500])


pred_dt[0] = 0

pred_dt[1] = 0

pred_dt[2] = 0

acc_dt=metrics.accuracy_score(y_train[0:500],pred_dt)*100


print('----------------------------------')

print('Accuracy of Random Forest = ',acc_dt,' %')
print('----------------------------------')



