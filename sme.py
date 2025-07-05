import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
df=pd.read_csv('sme.csv')
print(df)
print(df.info())
print(df.isnull().sum().sum())
df=df.drop(columns=['post_type','post_time','post_day'])
print(df)
le=LabelEncoder()
df['sentiment_score']=le.fit_transform(df['sentiment_score'])
print(df)
le=LabelEncoder()
df['platform']=le.fit_transform(df['platform'])
print(df)
mms=MinMaxScaler()
df['likes']=mms.fit_transform(df[['likes']])
mms1=MinMaxScaler()
df['comments']=mms1.fit_transform(df[['comments']])
mms2=MinMaxScaler()
df['shares']=mms2.fit_transform(df[['shares']])
print(df)
x=df[['platform','likes','comments','shares']]
y=df['sentiment_score']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
result = accuracy_score(y_test, y_pred)
print(result)
pickle.dump(model, open('sme_model.pkl', 'wb'))
# a = int(input("Enter Platform:"))
# b = int(input("Enter Likes:"))
# c = int(input("Enter Comments:"))
# d = int(input("Enter Shares:"))
# p = model.predict([[a, b, c, d]])
# print(p)
# # Save the model to a file
# model_filename = 'sme_model.pkl'
