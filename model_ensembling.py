import numpy as np
import pandas as pd

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

m1 = 'models/CNN/vgg_4/folded.csv'
m2 = 'submissions/pavel41/model_k0e6v0.991t0.969.csv'

df1 = pd.read_csv(m1)
arr_one = df1[list_classes]

df2 = pd.read_csv(m2)
arr_two = df2[list_classes]

a = np.corrcoef(x=arr_one,y=arr_two,rowvar=False)
print(np.mean(a))