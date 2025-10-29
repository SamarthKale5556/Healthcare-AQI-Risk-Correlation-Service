import pandas as pd
from db_connect import get_connection

# Load your CSVs
aqi_df = pd.read_csv("output/pune_aqi_12mo_synthetic.csv")
health_df = pd.read_csv("output/pune_health_synthetic.csv")
joined_df = pd.read_csv("output/pune_aqi_health_joined.csv")

# Connect to DB
conn = get_connection()
if conn:
    cur = conn.cursor()

    # Insert AQI data
    for _, row in aqi_df.iterrows():
        cur.execute("INSERT INTO aqi_data (date, aqi) VALUES (%s, %s)", (row['date'], row['AQI']))

    # Insert healthcare data
    for _, row in health_df.iterrows():
        cur.execute("INSERT INTO health_data (date, respiratory_cases) VALUES (%s, %s)", (row['date'], row['respiratory_cases']))

    # Insert joined data
    for _, row in joined_df.iterrows():
        cur.execute("INSERT INTO aqi_health_joined (date, aqi, respiratory_cases) VALUES (%s, %s, %s)",
                    (row['date'], row['AQI'], row['respiratory_cases']))

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Data inserted successfully into PostgreSQL!")
else:
    print("❌ Could not connect to database.")
