import pandas as pd
import numpy as np

#index = pd.date_range('1/1/2017', periods=100)

cate_id=np.random.randint(0,5, 30)
ts=pd.Series(np.random.normal(0.5,2,30), cate_id)

print ts


key=lambda x : x
zscore = lambda x:(x-x.mean())/ x.std()
trans = ts.groupby(key).transform(zscore)
print trans

zscore = lambda x:x.mean()
trans = ts.groupby(key).transform(zscore)
print trans


zscore = lambda x:x.std()
trans = ts.groupby(key).transform(zscore)
print trans
