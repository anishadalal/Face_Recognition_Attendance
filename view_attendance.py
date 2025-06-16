import sqlite3

conn = sqlite3.connect('attendance.db')
c = conn.cursor()

c.execute("SELECT * FROM attendance ORDER BY date DESC, time DESC")
rows = c.fetchall()

print("\n--- Attendance Records ---")
for row in rows:
    print(f"ID: {row[0]} | Date: {row[1]} | Time: {row[2]}")

conn.close()
