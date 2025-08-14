# ML_Projects
Practical approach to ML


Prepare the virtual environment


1)

pip install pandas numpy scikit-learn matplotlib seaborn jupyter

2) load the dataset

import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.shape)
train.head()

3) explore the data

Use .info() and .describe() to understand features.

Plot:

import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(train['SalePrice'], kde=True)
plt.show()

4. Preprocess
Handle missing values with fillna() or SimpleImputer.

Encode categorical variables with pd.get_dummies() or OneHotEncoder.

Normalize numeric columns with StandardScaler.

