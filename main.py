#Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from plotnine import *

#Reading the Data
movie_df=pd.read_csv("./data/movie_data.csv")

#Data Removal
movie_df.drop('movie_imdb_link', axis=1, inplace=True)
movie_df["color"].value_counts()
movie_df.drop('color',axis=1,inplace=True)
movie_df.dropna(axis=0,subset=['director_name', 'num_critic_for_reviews',
                               'duration','director_facebook_likes','actor_3_facebook_likes',
                               'actor_2_name','actor_1_facebook_likes','actor_1_name','actor_3_name',
                               'facenumber_in_poster','num_user_for_reviews','language','country',
                               'actor_2_facebook_likes','plot_keywords'],inplace=True)
movie_df.drop_duplicates(inplace=True)
movie_df.drop('language',axis=1,inplace=True)

#Replace
movie_df["content_rating"].fillna("R", inplace = True)
movie_df["aspect_ratio"].fillna(movie_df["aspect_ratio"].median(),inplace=True)
movie_df["budget"].fillna(movie_df["budget"].median(),inplace=True)
movie_df['gross'].fillna(movie_df['gross'].median(),inplace=True)

#Create
movie_df["Profit"]=movie_df['budget'].sub(movie_df['gross'], axis = 0)
movie_df['Profit_Percentage']=(movie_df["Profit"]/movie_df["gross"])*100

#Create value counts by country, filter to only 3 (USA, other, UK)
value_counts=movie_df["country"].value_counts()
vals = value_counts[:2].index

#Checking for the movies released year wise
print((ggplot(movie_df)
 + aes(x='title_year')
 + geom_bar(size=20)
))

#Relationship between the imdb score and the profit made by the movie
print((ggplot(movie_df, aes(x='imdb_score', y='Profit')) +
 geom_point() +
 stat_smooth(colour='blue', span=1)
))

#Checking for the imdb rating of the movies and compared with the countries
print((ggplot(movie_df, aes(x='country', y='imdb_score')) +
 geom_boxplot()
))

#Finding the correlation between imdb_rating with respect to no of facebook likes
print((ggplot(movie_df)
 + aes(x='imdb_score', y='movie_facebook_likes')
 + geom_point()
 + labs(title='IMDB_Score vs. Facebook like for Movies', x='IMDB scores', y='Facebook Likes for movies')
 + stat_smooth(colour='blue', span=1)
))


#Top 20 movies based on the profit they made
plt.figure(figsize=(10,8))
# Sorting the dataframe based on Profit and getting top 20
movie_df = movie_df.sort_values(by ='Profit', ascending=False)
movie_df_new = movie_df.head(20)

# Plotting
ax = sns.barplot(x='Profit', y='movie_title', data=movie_df_new, palette='viridis')
plt.xlabel('Profit')
plt.ylabel('Movie Title')
plt.title('Top 20 movies based on profit')
plt.tight_layout()
plt.show()

# Top 20 movies based on the profit percentage
plt.figure(figsize=(10,8))
movie_df = movie_df.sort_values(by ='Profit_Percentage' , ascending=False)
movie_df_new = movie_df.head(20)
ax = sns.pointplot(x='Profit_Percentage', y='budget', hue='movie_title', data=movie_df_new)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()

# Top 20 directors based on the IMDB ratings
plt.figure(figsize=(12,10))
movie_df= movie_df.sort_values(by ='imdb_score' , ascending=False)
movie_df_new=movie_df.head(20)
ax=sns.pointplot(x='director_name', y='imdb_score', hue='movie_title', data=movie_df_new)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()

#Commercial success vs critial acclaim
movie_df= movie_df.sort_values(by ='Profit_Percentage' , ascending=False)
movie_df_new=movie_df.head(20)
print((ggplot(movie_df_new)
 + aes(x='imdb_score', y='gross',color = "content_rating")
 + geom_point()
 +  geom_hline(aes(yintercept = 600)) +
  geom_vline(aes(xintercept = 10)) +
  xlab("Imdb score") +
  ylab("Gross money earned in million dollars") +
  ggtitle("Commercial success Vs Critical acclaim") +
  annotate("text", x = 8.5, y = 700, label = "High ratings \n & High gross")))

#Top 20 actors of movies based on the imdb rating of the movies
plt.figure(figsize=(10,8))
movie_df = movie_df.sort_values(by ='imdb_score' , ascending=False)
movie_df_new=movie_df.head(20)
ax=sns.pointplot(x='actor_1_name', y='imdb_score', hue='movie_title', data=movie_df_new)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()

# Country of Top 20 movies based on imdb rating
plt.figure(figsize=(10,8))
movie_df= movie_df.sort_values(by ='imdb_score' , ascending=False)
movie_df_new=movie_df.head(20)
ax=sns.pointplot(x='country', y='imdb_score', hue='movie_title', data=movie_df_new)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()

#Remove Data for Machine Learning:
movie_df.drop('director_name', axis=1, inplace=True)
movie_df.drop('actor_1_name',axis=1,inplace=True)
movie_df.drop('actor_2_name',axis=1,inplace=True)
movie_df.drop('actor_3_name',axis=1,inplace=True)
movie_df.drop('movie_title',axis=1,inplace=True)
movie_df.drop('plot_keywords',axis=1,inplace=True)
movie_df.drop('genres',axis=1,inplace =True)
movie_df.drop('Profit',axis=1,inplace=True)
movie_df.drop('Profit_Percentage',axis=1,inplace=True)

# Select only numeric columns
numeric_cols = movie_df.select_dtypes(include=[np.number])
corr = numeric_cols.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(13,7))

# create a mask, so we only see the correlation values once
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr, mask=mask, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
plt.show()

movie_df["imdb_binned_score"]=pd.cut(movie_df['imdb_score'], bins=[0,4,6,8,10], right=True, labels=False)+1
movie_df.drop('imdb_score',axis=1,inplace=True)
movie_df.head(5)
movie_df = pd.get_dummies(data = movie_df, columns = ['country'] , prefix = ['country'] , drop_first = True)
movie_df = pd.get_dummies(data = movie_df, columns = ['content_rating'] , prefix = ['content_rating'] , drop_first = True)

#Splitting the frame
X=pd.DataFrame(columns=['duration','director_facebook_likes','actor_1_facebook_likes','gross','num_voted_users','facenumber_in_poster','budget','title_year','aspect_ratio','movie_facebook_likes','Other_actor_facebbok_likes','critic_review_ratio','country_USA','country_other','content_rating_G','content_rating_GP','content_rating_M','content_rating_NC-17','content_rating_Not Rated','content_rating_PG','content_rating_PG-13','content_rating_Passed','content_rating_R','content_rating_TV-14','content_rating_TV-G','content_rating_TV-PG','content_rating_Unrated','content_rating_X'],data=movie_df)
y=pd.DataFrame(columns=['imdb_binned_score'],data=movie_df)
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=100)

# Handle NaNs
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Logistic Regression
logit = LogisticRegression(max_iter=1000)
logit.fit(X_train, np.ravel(y_train, order='C'))
y_pred = logit.predict(X_test)

#Confusion matrix for logistic regression**
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#KNN
knn = KNeighborsClassifier(n_neighbors=22)
knn.fit(X_train, np.ravel(y_train,order='C'))
knnpred = knn.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, knnpred)
print(cnf_matrix)
print("Accuracy:",metrics.accuracy_score(y_test, knnpred))

#SVC
svc= SVC(kernel = 'sigmoid')
svc.fit(X_train, np.ravel(y_train,order='C'))
svcpred = svc.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, svcpred)
print(cnf_matrix)
print("Accuracy:",metrics.accuracy_score(y_test, svcpred))

#Naive bayes
gaussiannb= GaussianNB()
gaussiannb.fit(X_train, np.ravel(y_train,order='C'))
gaussiannbpred = gaussiannb.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, gaussiannbpred)
print(cnf_matrix)
print("Accuracy:",metrics.accuracy_score(y_test, gaussiannbpred))

#Decision Tree
dtree = DecisionTreeClassifier(criterion='gini') #criterion = entopy, gini
dtree.fit(X_train, np.ravel(y_train,order='C'))
dtreepred = dtree.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, dtreepred)
print(cnf_matrix)
print("Accuracy:",metrics.accuracy_score(y_test, dtreepred))

#Ada Boosting
abcl = AdaBoostClassifier(estimator=dtree, n_estimators=60)
abcl=abcl.fit(X_train,np.ravel(y_train,order='C'))
abcl_pred=abcl.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, abcl_pred)
print(cnf_matrix)
print("Accuracy:",metrics.accuracy_score(y_test, abcl_pred))

#Random Forest
rfc = RandomForestClassifier(n_estimators = 200)#criterion = entopy,gini
rfc.fit(X_train, np.ravel(y_train,order='C'))
rfcpred = rfc.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, rfcpred)
print(cnf_matrix)
print("Accuracy:",metrics.accuracy_score(y_test, rfcpred))

#Bagging classfier
new_movie_df=movie_df.pop("imdb_binned_score")
bgcl = BaggingClassifier(n_estimators=60, max_samples=.7 , oob_score=True)
bgcl = bgcl.fit(movie_df, new_movie_df)
print(bgcl.oob_score_)

#Gradient boosting
gbcl = GradientBoostingClassifier(n_estimators = 50, learning_rate = 0.09, max_depth=5)
gbcl = gbcl.fit(X_train,np.ravel(y_train,order='C'))
test_pred = gbcl.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, test_pred)
print(cnf_matrix)
print("Accuracy:",metrics.accuracy_score(y_test, test_pred))

#xg boosting
y_train = y_train - 1
y_test = y_test - 1
xgb = XGBClassifier()
xgb.fit(X_train, np.ravel(y_train,order='C'))
xgbprd = xgb.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, xgbprd)
print(cnf_matrix)
print("Accuracy:",metrics.accuracy_score(y_test, xgbprd))

print('Logistic  Reports\n',classification_report(y_test, y_pred, zero_division=0))
print('KNN Reports\n',classification_report(y_test, knnpred, zero_division=0))
print('SVC Reports\n',classification_report(y_test, svcpred, zero_division=0))
print('Naive BayesReports\n',classification_report(y_test, gaussiannbpred, zero_division=0))
print('Decision Tree Reports\n',classification_report(y_test, dtreepred, zero_division=0))
print('Ada Boosting\n',classification_report(y_test, abcl_pred, zero_division=0))
print('Random Forests Reports\n',classification_report(y_test, rfcpred, zero_division=0))
print('Gradient Boosting',classification_report(y_test, test_pred, zero_division=0))
print('XGBoosting\n',classification_report(y_test, xgbprd, zero_division=0))

