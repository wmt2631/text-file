import pandas as pd 
import numpy as np 
import matplotlib

dates = pd.date_range('20130101',periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index= dates,columns=['A','B','C','D'])
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan

print(df,end='\n')
#print(df.dropna(axis=0,how='any'))
#how=['any','all']  axis=0 行 axis=1 列
#print(df.fillna(value=0)) #缺失的数据变为0
#print(df.isnull()) 是否有数据缺失
print(np.any(df.isnull()) == True ) #是否有数据缺失
