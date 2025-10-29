import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from db_connect import get_connection

# Connect to PostgreSQL
conn = get_connection()
query = "SELECT date, aqi, respiratory_cases FROM aqi_health_joined ORDER BY date;"
df = pd.read_sql(query, conn)

# --- Plot 1: AQI & Respiratory Cases Over Time ---
plt.figure(figsize=(10,5))
plt.plot(df['date'], df['aqi'], label='AQI', linewidth=2)
plt.plot(df['date'], df['respiratory_cases'], label='Respiratory Cases', linewidth=2)
plt.title('AQI vs Respiratory Cases (Pune)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 2: Scatter Correlation ---
plt.figure(figsize=(6,6))
plt.scatter(df['aqi'], df['respiratory_cases'], alpha=0.6)
plt.title('Correlation between AQI and Respiratory Cases')
plt.xlabel('AQI')
plt.ylabel('Respiratory Cases')
plt.tight_layout()
plt.show()

conn.close()
