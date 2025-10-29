**🌍 Healthcare–AQI Risk Correlation Service**


**📘 Overview**

This project builds a data engineering pipeline that connects real-time air quality data with healthcare analytics.
It fetches live AQI data from the OpenAQ API, cleans and stores it in PostgreSQL, integrates it with synthetic healthcare data, and analyzes how pollution impacts respiratory illnesses.


**🎯 Objective**

**To design a data pipeline that automatically:**

**Fetches live AQI data for Pune**

**Cleans and stores it in PostgreSQL**

**Integrates healthcare (respiratory case) data**

**Finds correlations and visualizes results**


**⚙️ Workflow**


Stage	Description	Files


**1️⃣	Data Capture** – Fetch live AQI using OpenAQ API	fetch_pune_air_quality_v3.py

**2️⃣	Data Cleaning** – Store and clean AQI data	cleaned/live_pune_aqi.csv

**3️⃣	Data Processing** – Create 12-month realistic AQI dataset	produce_aqi_health_pipeline.py

**4️⃣	Database Storage** – Insert cleaned data into PostgreSQL	db_connect.py, insert_to_postgre.py

**5️⃣	Visualization**– Show AQI–health trends and lag impact	visualize_postgres_data.py



**🧩 Tools & Technologies Used**


1.Python (Data collection, cleaning, analysis)

2.PostgreSQL (Data storage and querying)

3.Pandas, NumPy, Matplotlib (Processing & visualization)

4.Scikit-learn, SciPy (Statistical correlation)

5.OpenAQ API v3 (Live AQI data source)


**📊 Key Results**

Found a strong positive correlation (r = 0.646) between AQI and respiratory illness cases.

Health impact usually lags by 4–5 days after pollution peaks.

All data stored securely in PostgreSQL and visualized as interactive charts.


**🌍 Real-World Applications**

Public health monitoring and policy-making

Pollution impact prediction and awareness

Smart city and healthcare analytics dashboards

**🧠 Summary**

A complete Python + PostgreSQL data engineering pipeline that links air quality trends with healthcare data, proving how real-time data can improve public health insights and environmental awareness.
