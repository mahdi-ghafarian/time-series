import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv("pairplot\\data.csv")
df = df.drop(["YearMonth","Labour Force","Not Employed"], axis=1)

# Set color palette
sns.set_palette('YlGnBu')
# sns.set_palette('Blues')
# sns.set_palette(sns.cubehelix_palette(start=0.5, rot=-0.75, dark=0.3, light=0.8, reverse=False))

# Variables
x='Population'
y='Employment'

# (1) Scatter plot with hue and regression line (uncomment both lines)
sns.scatterplot(data=df,hue='Decade', x=x,y=y, s=100, alpha=0.75)
sns.regplot(data=df,x=x,y=y,scatter=False, color='red')

# (2) Simple regression plot for two variables
# sns.regplot(data=df,x=x,y=y)

# (3) Pairplot with hue
# sns.pairplot(df, hue='Decade')

# (4) Pairplot with regression line
# sns.pairplot(df, kind='reg', plot_kws={'line_kws': {'color': 'red'}})


# Show the plot
plt.show()

