# unit13-challenge

## Option 2: Clustering Crypto

###  Initial imports
```
import requests
import pandas as pd
import matplotlib.pyplot as plt
import hvplot.pandas
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pathlib import Path
```

### Fetching Cryptocurrency Data
```
file_path = Path("Resources/crypto_data.csv")

# Create a DataFrame
crypto_df = pd.read_csv(file_path)
crypto_df.head(20)
```
pic data

### Data Preprocessing

#### drop Unnamed: 0
```
crypto_df.drop(columns="Unnamed: 0", inplace = True)
crypto_df
```
![](https://github.com/bleachevil/unit13-challenge/blob/main/pic/data.png?raw=true)


#### Keep only cryptocurrencies that are trading and drop IsTrading
```
index_names = crypto_df[ crypto_df['IsTrading'] == False ].index
crypto_df.drop(index_names, inplace = True)
crypto_df.reset_index(inplace=True)
crypto_df.drop(columns="index", inplace = True)
crypto_df.drop(columns="IsTrading", inplace = True)
crypto_df
```
!](https://github.com/bleachevil/unit13-challenge/blob/main/pic/dropistrading.png?raw=true)

#### drop NA & Null and remove rows with cryptocurrencies having no coins mined
```
crypto_df.dropna(inplace=True)
crypto_df = crypto_df.drop(crypto_df[crypto_df.TotalCoinsMined == 0].index)
crypto_df
```
![](https://github.com/bleachevil/unit13-challenge/blob/main/pic/dropnocoin.png?raw=true)

#### Store the 'CoinName'column in its own DataFrame prior to dropping it from crypto_df
```
coinname = crypto_df[['CoinName']]
coinname.reset_index(inplace=True)
coinname.drop(columns = "index", inplace= True)
coinname
```
![](https://github.com/bleachevil/unit13-challenge/blob/main/pic/coinname.png?raw=true)

#### Drop the 'CoinName' and create X and dummy variables for text features
```
X = crypto_df.drop(columns = "CoinName")
X['TotalCoinSupply'] = pd.to_numeric(X['TotalCoinSupply'])
X = pd.get_dummies(X,columns=["Algorithm","ProofType"])
X
```
![](https://github.com/bleachevil/unit13-challenge/blob/main/pic/X.png?raw=true)

#### Standardize data
```
iris_scaled = StandardScaler().fit_transform(X)
print(iris_scaled[0:5])
```
![](https://github.com/bleachevil/unit13-challenge/blob/main/pic/standardize.png?raw=true)

### Reducing Dimensions Using PCA
```
pca = PCA(n_components=3)
iris_pca = pca.fit_transform(iris_scaled)
df_iris_pca = pd.DataFrame(data=iris_pca, columns=["PC 1", "PC 2","PC 3"])
pcs_df = pd.concat([coinname,df_iris_pca], axis= 'columns')
pcs_df.set_index("CoinName",inplace=True)
pcs_df
```
![](https://github.com/bleachevil/unit13-challenge/blob/main/pic/pcs.png?raw=true)

### Clustering Crytocurrencies Using K-Means

#### Find the Best Value for k Using the Elbow Curve
```
inertia = []
k = list(range(1, 11))

# Calculate the inertia for the range of k values
for i in k:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(X)
    inertia.append(km.inertia_)

# Create the Elbow Curve using hvPlot
elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)
df_elbow.hvplot.line(x="k", y="inertia", title="Elbow Curve", xticks=k)
```
![](https://github.com/bleachevil/unit13-challenge/blob/main/pic/elbow.png?raw=true)

#### Running K-Means with k= 4
```
# Initialize the K-Means model
model = KMeans(n_clusters=4, random_state=5)
# Fit the model
model.fit(pcs_df)
# Predict clusters
predictions = model.predict(pcs_df)
# Create a new DataFrame including predicted clusters and cryptocurrencies features
pcs_df["class"] = model.labels_
pcs_df.head()
```
![](https://github.com/bleachevil/unit13-challenge/blob/main/pic/kmean.png?raw=true)

### Visualizing Results

#### Scatter Plot with Tradable Cryptocurrencies
```
crypto_df.set_index("CoinName",inplace=True)
clustered_df = pd.concat([crypto_df,pcs_df], axis= 'columns')
clustered_df['TotalCoinSupply'] = pd.to_numeric(clustered_df['TotalCoinSupply'])
clustered_df
```
![](https://github.com/bleachevil/unit13-challenge/blob/main/pic/clustered.png?raw=true)
```
clustered_df.hvplot.scatter(
    x="TotalCoinsMined",
    y="TotalCoinSupply",
    by="class",
    hover_cols=["CoinName"])
```
![](https://github.com/bleachevil/unit13-challenge/blob/main/pic/plot.png?raw=true)

#### Table of Tradable Cryptocurrencies
```
clustered_df.hvplot.table(columns=["CoinName", "Algorithm", "ProofType", "TotalCoinSupply", "TotalCoinsMined", "class"], sortable=True, selectable=True)
```

![](https://github.com/bleachevil/unit13-challenge/blob/main/pic/table.png?raw=true)
```
clustered_df.index.value_counts()
```
![](https://github.com/bleachevil/unit13-challenge/blob/main/pic/count.png?raw=true)





