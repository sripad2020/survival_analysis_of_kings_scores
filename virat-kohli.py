import matplotlib.pyplot as plt
import numpy
import pandas as pd
data=pd.read_csv('71 Centuries of Virat Kohli.csv')
print(' The description of the columns ')
print(data.describe())
cat=data.select_dtypes(include='object').columns
print('The categorical columns are ')
print(cat)
num=data.select_dtypes(include=numpy.number).columns
print('The numerical column are ')
print(num)
nan_values=data.isnull().sum()
print(nan_values)
import seaborn as sn
sn.pairplot(data)
plt.show()

plt.plot(data['Score'],data['Batting Order'])
plt.xlabel('score of King ')
plt.ylabel('Batting order of King')
plt.legend()
plt.show()

from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
data['captain']=lab.fit_transform(data["Captain"])
data['result']=lab.fit_transform(data['Result'])
data['format']=lab.fit_transform(data['Format'])
data['out/not_out']=lab.fit_transform(data['Out/Not Out'])
data['mom']=lab.fit_transform(data['Man of the Match'])
data['date']=lab.fit_transform(data['Date'])

sn.clustermap(data[['captain','result','format','out/not_out','Score','Batting Order','mom']])
plt.show()

sn.countplot(data['mom'])
plt.xlabel('No of Man of the Match awards are ')
plt.legend()
plt.show()

sn.countplot(data['captain'])
plt.xlabel('captain wins ')
plt.legend()
plt.show()

sn.scatterplot(data['Score'],data['Batting Order'])
plt.xlabel('score ')
plt.ylabel('batting order')
plt.legend()
plt.show()

plt.plot(data['captain'],data['result'])
plt.xlabel('caption ')
plt.ylabel('result of captaincy')
plt.legend()
plt.show()

plt.plot(data['Score'],data['captain'])
plt.xlabel('score ')
plt.ylabel('captain ')
plt.legend()
plt.show()

from lifelines import KaplanMeierFitter
x=data['Score']
y=data['result']
km=KaplanMeierFitter()
km.fit(x,y)
km.plot()
plt.show()
print('The median survival time of king-kohli is ')
print(km.median_survival_time_)
print(km.confidence_interval_)
X=data['date']
Y=data['Score']



plt.plot(data['Score'], label="score of king", color="black")
plt.plot(data['date'], label="dates of scored ", color="red")
plt.xlabel("Score of king")
plt.ylabel("dates of king score")
plt.legend()
plt.show()