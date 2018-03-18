import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#data = pd.Series(np.random.randn(1000),index=np.arange(1000))
#data = data.cumsum()

data = pd.DataFrame(np.random.randn(1000,4),
	index=np.arange(1000),
	columns=list("ABCD"))
data = data.cumsum()
data.plot()
plt.show()