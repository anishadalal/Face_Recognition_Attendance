import cv2
import sqlite3
from datetime import datetime

# Connect to SQLite database
conn = sqlite3.connect('attendance.db')
c = conn.cursor()

# Create table if not exists
c.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER,
        date TEXT,
        time TEXT
    )
''')

# Load recognizer and face cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)
marked = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])

        if conf < 50 and id not in marked:
            now = datetime.now()
            c.execute("INSERT INTO attendance (id, date, time) VALUES (?, ?, ?)", 
                      (id, now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')))
            conn.commit()
            marked.add(id)
            print(f"✅ Attendance marked for ID: {id}")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face Recognition Attendance', frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
conn.close()
cv2.destroyAllWindows()
