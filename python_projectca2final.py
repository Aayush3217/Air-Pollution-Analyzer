import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("D:/Python Project/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69.csv")

# OBJECTIVE 1: Data Cleaning & Preprocessing
df.drop_duplicates(inplace=True)
df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')
df = df.dropna(subset=['pollutant_avg'])
print(df)


# OBJECTIVE 2: Top Polluted Cities
top_polluted = (
    df.groupby('city')['pollutant_avg']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_polluted.values, y=top_polluted.index, palette="Reds_r")
plt.title("Top 10 Most Polluted Cities (by Avg Pollutant Level)")
plt.xlabel("Average Pollutant Level")
plt.tight_layout()
plt.show()
plt.close()

# OBJECTIVE 3: Pollution by Pollutant Type
pollutant_stats = df.groupby('pollutant_id')['pollutant_avg'].mean().sort_values()
plt.figure(figsize=(8, 6))
sns.barplot(x=pollutant_stats.values, y=pollutant_stats.index, palette="Blues_d")
plt.title("Average Pollution by Pollutant Type")
plt.xlabel("Average Level")
plt.tight_layout()
plt.show()
plt.close()

# OBJECTIVE 4: Geospatial Visualization
map_data = df[['latitude', 'longitude', 'pollutant_avg']].dropna()
plt.figure(figsize=(10, 6))
sc = plt.scatter(
    map_data['longitude'], map_data['latitude'],
    c=map_data['pollutant_avg'], cmap='Reds', alpha=0.7, marker='o'
)
plt.colorbar(sc, label='Pollutant Level')
plt.title("Geospatial Pollution Visualization")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()
plt.close()

# OBJECTIVE 5: Multi-variable Regression (Latitude + Longitude)
X = df[['latitude', 'longitude']]
y = df['pollutant_avg']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.5, color='purple')
plt.xlabel("Actual Pollutant Avg")
plt.ylabel("Predicted Pollutant Avg")
plt.title("Multi-variable Linear Regression")
plt.tight_layout()
plt.show()

# OBJECTIVE 6: Pollutant Level Category Distribution
def categorize_pollution(value):
    if value <= 50:
        return 'Low'
    elif value <= 100:
        return 'Moderate'
    else:
        return 'High'

df['pollution_category'] = df['pollutant_avg'].apply(categorize_pollution)
category_counts = df['pollution_category'].value_counts()
plt.figure(figsize=(6, 6))
category_counts.plot.pie(autopct='%1.1f%%', colors=['green', 'orange', 'red'])
plt.title('Pollution Level Category Distribution')
plt.ylabel('')
plt.tight_layout()
plt.show()
plt.close()