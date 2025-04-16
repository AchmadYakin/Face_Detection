import cv2

face_ref = cv2.CascadeClassifier("face_ref.xml")

camera = cv2.VideoCapture(0)

while True :
    _, frame = camera.read()
    
    optimaze = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Mendeteksi wajah dalam frame
    faces = face_ref.detectMultiScale(
        optimaze,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(30, 30)
    )

    # Menandai wajah yang terdeteksi dengan kotak dan teks
    for i, (x, y, w, h) in enumerate(faces):
        # Tambahkan margin untuk crop
        margin_x = int(w * 0.3)
        margin_y = int(h * 0.4)

        x1 = max(x - margin_x, 0)
        y1 = max(y - margin_y, 0)
        x2 = min(x + w + margin_x, frame.shape[1])
        y2 = min(y + h + margin_y, frame.shape[0])

        # 👉 Crop wajah dulu dari frame ASLI
        face_crop = frame[y1:y2, x1:x2].copy()  # gunakan .copy() untuk jaga aman

        # 👉 Setelah crop, baru tambahkan rectangle dan text ke frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(frame, "Wajah", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Tampilkan wajah hasil crop (tanpa rectangle)
        cv2.imshow(f"Wajah {i+1}", face_crop)


    # Menampilkan hasil deteksi di jendela
    cv2.imshow("Face Detection", frame)

    # Menutup jendela jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Membersihkan resource
camera.release()
cv2.destroyAllWindows()