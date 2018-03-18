import pandas as pd 
import numpy as np 
#import matplotlib

dates = pd.date_range('20130101',periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index= dates,columns=['A','B','C','D'])

print(df,end='\n')
#print(df,df['A'],df.A) 列
#print(df[0:3]) 行
#print(df['2013-01-02':'2013-01-04'])
#print (df.loc['20130102']) 20130102行的信息
#print(df.loc['2013-01-02',['A','B']])
#print(df.iloc[3:5,1:3]) 3-5行1-3列
#print (df.iloc[[1,3,5],1:3])
#print(df.ix[:3,['A','B']])
print (df[df.A>8])