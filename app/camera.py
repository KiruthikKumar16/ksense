import cv2

def get_webcam_frame():
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
        else:
            return None
    except Exception:
        return None 