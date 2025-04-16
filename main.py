import cv2

# menggunakan Algoritma Haar Cascade dengan data face_ref.xml
face_ref = cv2.CascadeClassifier("face_ref.xml")

# mendeklarasi variable kamera. 0 untuk kamera bawaan
camera = cv2.VideoCapture(0)

# face detection
def face_detection(open_cam):
    # mengubah camera menjadi gray
    optimaze = cv2.cvtColor(open_cam, cv2.COLOR_RGB2BGR)
    # mendeteksi Gambar
    faces = face_ref.detectMultiScale(optimaze, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30) )
    return faces

# kotak untuk melabelkan wajah
def label(open_cam):
    # x = horizontal, w = vertikal, w = width, h = height
    for x, y, w, h in face_detection(open_cam):
        # membuat kotak
        cv2.rectangle(open_cam, (x,y), (x + w , y + h), (0,0,255), 4)
        # membuat text tulisan wajah
        cv2.putText(open_cam, "Wajah", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

def close_windows() :
    camera.release()
    cv2.destroyAllWindows()
    exit()

def main() :
    while True :
        _, open_cam = camera.read()
        label(open_cam)
        cv2.imshow("Face Detection", open_cam)

        # close windows dengan "q"
        if cv2.waitKey(1) & 0xFF == ord('q') :
            close_windows()

if __name__ == '__main__' :
    main()