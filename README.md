🌍 Healthcare–AQI Risk Correlation Service
📘 Overview

This project builds a data engineering pipeline that connects real-time air quality data (AQI) with healthcare analytics.
It fetches live AQI data from the OpenAQ API, cleans and stores it in PostgreSQL, integrates it with synthetic healthcare data, and analyzes how pollution impacts respiratory illnesses.

🎯 Objective

To design a pipeline that automatically:

Fetches live AQI data for Pune

Cleans and stores it in PostgreSQL

Integrates healthcare (respiratory case) data

Finds correlations and visualizes results

⚙️ Workflow
Stage	Description	Files
1️⃣ Data Capture	Fetch live AQI using OpenAQ API	fetch_pune_air_quality_v3.py
2️⃣ Data Cleaning	Store and clean data	cleaned/live_pune_aqi.csv
3️⃣ Data Processing	Create 12-month realistic AQI dataset	produce_aqi_health_pipeline.py
4️⃣ Database Storage	Insert cleaned data into PostgreSQL	db_connect.py, insert_to_postgre.py
5️⃣ Visualization	Show AQI–health trends and lag impact	visualize_postgres_data.py
🧩 Tools & Technologies Used

🐍 Python

🗄️ PostgreSQL

📊 Pandas, NumPy, Matplotlib

🤖 Scikit-learn, SciPy

🌫️ OpenAQ API (v3)

📊 Key Results

Strong positive correlation (r = 0.646) between AQI and respiratory cases

Health impact usually lags 4–5 days after pollution peaks

All data stored securely in PostgreSQL and visualized using charts

🌍 Real-World Applications

🏥 Public Health Monitoring – Predict hospital admissions due to pollution

🌫️ Environmental Analysis – Identify pollution trends and patterns

🧑‍⚕️ Policy Making – Support data-driven government action

🏙️ Smart City Dashboards – Integrate with urban monitoring systems

🏁 Summary

A complete Python + PostgreSQL Data Engineering pipeline that connects real-time AQI data with healthcare insights.
This system demonstrates how data-driven analysis can help monitor public health risks, improve environmental awareness, and support decision-making for cleaner and healthier cities.
