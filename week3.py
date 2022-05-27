import inline as inline
import matplotlib
import pandas as pand
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import inline as inl

dataFrame = pand.read_csv("C:\\Users\\START\\Downloads\\module_5_auto.csv", )
# print(dataFrame[['bore','stroke','compression-ratio','horsepower']].corr())

# print(dataFrame[["engine-location", 'price']].corr())
# sns.regplot(x="engine-location", y="price", data=dataFrame)
# plt.ylim(0,)
# plt.show()

# sns.boxplot(x="engine-location", y="price", data=dataFrame)
# plt.ylim(0,)
# plt.show()

# engines = dataFrame['engine-location'].value_counts().to_frame()
# engines.rename(columns={"engine-location" : "count"}, inplace=True)
# engines.index.name = "location"
# print(engines)

# body_drive = dataFrame[["body-style", 'drive-wheels', 'price']]
# stats = body_drive.groupby(["body-style", 'drive-wheels'], as_index=False).mean()\
#     .pivot(index='body-style', columns='drive-wheels').fillna(0)
# plt.pcolor(stats, cmap='RdBu')
# plt.colorbar()
# plt.show()

frameGroup = dataFrame[['drive-wheels','body-style','price']].groupby(['body-style'])
print(frameGroup.get_group('convertible')['price'].to_string())
