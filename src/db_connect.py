import psycopg2

def get_connection():
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="aqi_health_db",  # name from pgAdmin
            user="postgres",           # your pgAdmin username
            password="ADT23SOCB0955",  # 🔒 replace with your actual password
            port="5432"                # default PostgreSQL port
        )
        print("✅ Connected to PostgreSQL successfully!")
        return conn
    except Exception as e:
        print("❌ Database connection failed:", e)
        return None

if __name__ == "__main__":
    get_connection()
