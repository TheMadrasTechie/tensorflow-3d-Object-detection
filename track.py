import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

cap = cv2.VideoCapture(0)


# {'Shoe', 'Chair', 'Cup', 'Camera'}.
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=1,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5,
                            model_name='Cup') as objectron:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = objectron.process(image)
        height = image.shape[0]
        width = image.shape[1]
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.line(image, (int(width/2),0), (int(width/2),height), (255,0,0), 1)
        if results.detected_objects:            
            for detected_object in results.detected_objects:
                print(10)
                x = int(detected_object.landmarks_2d.landmark[0].x *width)
                y = int(detected_object.landmarks_2d.landmark[0].y *height)                                
                print("x "+str(x))
                print("y "+str(y))
                image = cv2.circle(image, (x,y), 5, (0,255,255), 2)
                mp_drawing.draw_landmarks(
                    image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
        cv2.imshow('MediaPipe Objectron', rescale_frame(image, percent=150))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()