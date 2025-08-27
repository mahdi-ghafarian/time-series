import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Historical S&P 500 annual returns from 1928 to 2024
returns_data = {
    1928: 43.81, 1929: -8.42, 1930: -24.90, 1931: -43.34, 1932: -8.19, 1933: 53.99, 1934: -1.44, 1935: 47.67, 1936: 33.92, 1937: -35.03,
    1938: 31.12, 1939: -0.41, 1940: -9.78, 1941: -11.59, 1942: 20.34, 1943: 25.90, 1944: 19.75, 1945: 36.44, 1946: -8.07, 1947: 5.71,
    1948: 5.50, 1949: 18.79, 1950: 31.71, 1951: 24.02, 1952: 18.37, 1953: -0.99, 1954: 52.62, 1955: 31.56, 1956: 6.56, 1957: -10.78,
    1958: 43.36, 1959: 11.96, 1960: 0.47, 1961: 26.64, 1962: -8.73, 1963: 22.80, 1964: 16.48, 1965: 12.45, 1966: -10.06, 1967: 23.98,
    1968: 11.06, 1969: -8.50, 1970: 4.01, 1971: 14.31, 1972: 18.76, 1973: -14.66, 1974: -26.47, 1975: 37.20, 1976: 23.84, 1977: -7.18,
    1978: 6.56, 1979: 18.44, 1980: 32.42, 1981: -4.91, 1982: 21.55, 1983: 22.56, 1984: 6.27, 1985: 31.73, 1986: 18.67, 1987: 5.25,
    1988: 16.61, 1989: 31.49, 1990: -3.10, 1991: 30.47, 1992: 7.62, 1993: 10.08, 1994: 1.32, 1995: 37.58, 1996: 22.96, 1997: 33.36,
    1998: 28.58, 1999: 21.04, 2000: -9.10, 2001: -11.89, 2002: -22.10, 2003: 28.68, 2004: 10.88, 2005: 4.91, 2006: 15.79, 2007: 5.49,
    2008: -37.00, 2009: 26.46, 2010: 15.06, 2011: 2.11, 2012: 16.00, 2013: 32.39, 2014: 13.69, 2015: 1.38, 2016: 11.96, 2017: 21.83,
    2018: -4.38, 2019: 31.49, 2020: 18.40, 2021: 28.71, 2022: -18.11, 2023: 26.29, 2024: 25.02
}

# Convert to DataFrame
df = pd.DataFrame(list(returns_data.items()), columns=['Year', 'Return'])
df['Decade'] = (df['Year'] // 10) * 10

# Group returns by decade
grouped = df.groupby('Decade')['Return'].apply(list)

# Create boxplot
fig, ax = plt.subplots(figsize=(12, 6))
box = ax.boxplot(grouped.values, patch_artist=True, widths=0.6)

# Customize boxplot appearance
for element in ['boxes', 'whiskers', 'caps']:
    plt.setp(box[element], color='black')
for median in box['medians']:
    median.set(color='red')
for patch in box['boxes']:
    patch.set(facecolor='none')

# Add mean circles
for i, decade_returns in enumerate(grouped.values):
    mean_val = np.mean(decade_returns)
    ax.plot(i + 1, mean_val, 'o', color='blue')

# Add horizontal lines
overall_mean = df['Return'].mean()
ax.axhline(y=overall_mean, color='blue', linestyle='--', label=f'Average: {overall_mean:.1f}%')
ax.axhline(y=0, color='gray', linestyle=':', label='')

# Set x-axis labels
ax.set_xticklabels([f"{int(decade)}s" for decade in grouped.index])
ax.set_ylabel('Annual Return (%)')
ax.set_title('S&P 500 Annual Returns by Decade (1928–2024)')
ax.legend()

plt.tight_layout()
# plt.savefig("./spx-return/sp500_returns_by_decade.png")
print("Boxplot saved as 'sp500_returns_by_decade.png'.")
plt.legend()
plt.show()