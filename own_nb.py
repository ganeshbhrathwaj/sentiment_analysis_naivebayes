#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

g="have a good day"

#importing_dataset
ds=pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t',quoting=3)

#cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
c=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',ds['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    c.append(review)

#creating a bag of words  
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(c).toarray()
y=ds.iloc[:,1].values


#cleaning my text
d=[]
txt=re.sub('[^a-zA-Z]',' ',g)
txt=txt.lower()
txt=txt.split()
ps=PorterStemmer()
txt=[ps.stem(word) for word in txt if not word in set(stopwords.words('english'))]
txt=' '.join(txt)
d.append(txt)

#baging my text
txt=txt.split()
z=np.zeros((1000,len(txt)))
for e in c:
    for q in txt:
        if(q in e):z[c.index(e)][txt.index(q)]=1
sg=0
nsg=0
pr=0
nr=0
s1=0
s2=0
lst=[]
lst1=[]
lst2=[]
lst3=[]

#to claculate probablitu of good and day in review (denominator)
for j in range(0,len(txt)):
    s1=0
    for i in range(0,1000):
        if(z[i][j]==1):s1+=1
        if(z[i][j]==0):s2+=1
    ps1=s1/1000
    ps2=s2/1000
    if(ps1==0):ps1=0.01
    lst.append(ps1)
    lst3.append(ps2)

#to calculate probablity of positive and and negative review(numerator)
for i in range(0,1000):    
    if(y[i]==1):pr=pr+1
    if(y[i]==0):nr=nr+1
ppr=pr/1000
pnr=nr/1000

#to calculate probablity of p(g/1)     
for j in range(0,len(txt)):
    sg=0
    nsg=0
    for i in range(0,1000):
        if(z[i][j]==1 and y[i]==1):sg=sg+1
        if(z[i][j]==1 and y[i]==0):nsg=nsg+1
    psg=sg/500
    nnsg=nsg/500
    if(psg==0):psg=100
    if(nnsg==0):nnsg=100
    lst1.append(psg)
    lst2.append(nnsg)

#to calculate p(1/g)
p0=p1=p2=p3=1
for m in lst:
    print(m)
    p0=p0*m

for m in lst1:
    p1=p1*m
    
for m in lst2:
    p2=p2*m
    
for m in lst3:
    p3=p3*m

pp=(p1*ppr)/(p0)
npp=(p2*pnr)/(p3)
pp=pp/(pp+npp)
npp=npp/(pp+npp)
print(pp)
print(npp)
if(pp>=npp):print("positive") 
else:print("negative") 
    
    
    
            
                
     
    
z=cv.fit_transform(d).toarray()