import pickle
model_file='sme_model.pkl'
model=pickle.load(open(model_file, 'rb'))
a = int(input("Enter Platform:"))
b = int(input("Enter Likes:"))
c = int(input("Enter Comments:"))
d = int(input("Enter Shares:"))
p = model.predict([[a, b, c, d]])
print(p)