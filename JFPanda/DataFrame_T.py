import pandas as pd
import numpy as np
from pandas import *
from numpy import *
df = pd.DataFrame(np.random.rand(5, 6), columns=list("abcdef"))
print(df.iloc[0, 0])
print(df.loc[0])

