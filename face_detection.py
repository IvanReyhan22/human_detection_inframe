import os
import numpy as np
import cv2

def main():
    # set direktori root file
    directory = os.path.dirname(__file__)
    # load gambar
    image_path = os.path.join(directory, "detect.jpeg")
    original_image = cv2.imread(image_path)

    # resize gambar ke resolusi 1000 x 750
    target_size = (1000, 750)
    resized_image = cv2.resize(original_image, target_size)

    # simpan gambar hasil resized ke file baru
    resized_image_path = os.path.join(directory, "resized_detect.jpeg")
    cv2.imwrite(resized_image_path, resized_image)

    # load gambar dalam bentuk video capture (image/camera)
    capture = cv2.VideoCapture(os.path.join(directory, "resized_detect.jpeg"))
    # capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        exit()
    
    # load model yunet face detection
    weights = os.path.join(directory, "yunet.onnx")
    # inisialisasi face detector dengan OpenCV
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))

    while True:
        result, image = capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        # channels = 1 if len(image.shape) == 2 else image.shape[2]
        # if channels == 1:
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # if channels == 4:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # resolusi gambar
        height, width, _ = image.shape

        # set parameter ukuran gambar
        face_detector.setInputSize((width, height))
        
        # proses pendeteksian wajah pada gambar
        _, faces = face_detector.detect(image)

        # pengecekan jika wajah terdeteksi
        faces = faces if faces is not None else []

        for index,face in enumerate(faces):
            # mengammbarkan kotak pada wajah yang terdeteksi
            box = list(map(int, face[:4]))
            # menentukan warna dan ketebalan kotak
            color = (0, 255, 255)
            thickness = 2
            # proses menggambarkan kotak ke gambar wajah
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

            # menuliskan urutan pendeteksian wajah
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 2
            cv2.putText(image, str(index + 1), position, font, scale, color, thickness, cv2.LINE_AA)

        cv2.imshow("face detection", image)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()