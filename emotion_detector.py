from tensorflow.keras.preprocessing.image import img_to_array
import config
from keras.models import load_model
import numpy as np
import imutils
import cv2

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#("C:\\Users\\flavi\\OneDrive\\Desktop\\ImageProcessing_ComputerVision\\Projects\\Training_HaarCascade_OpenCV\\cascade\\cascade.xml")
#('haarcascade_frontalface_default.xml')
model = load_model("outputs/model")
EMOTIONS = ["angry", "scared", "happy", "sad", "surprised", "neutral"]

camera = cv2.VideoCapture(0)

while True:
    (grabbed, frame) = camera.read()

    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #  build an empty canvas where we will plot the probabilities of all emotions
            #  in a histogram-like diagram where length of rectangle ~ probability
    canvas = np.zeros((220, 300, 3), dtype="uint8")

    frameClone = frame.copy()

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=12,
                                      minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(rects) > 0:
        rect = max(rects, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
                                #Take only the biggest face

        (fX, fY, fW, fH) = rect
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float")
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):

            text = "{}: {}".format(emotion, prob * 100)

            w = int(prob * 300)  # set the width of the rectangle ~ w the probability
            cv2.rectangle(canvas, (5, (i * 35) + 5),
                          (w, (i * 35) + 35), (0, 0, 255), -1)  # position it accordingly
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 2)

        cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 255), 2)  # this text is for the detected face
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
                                            # as well as this is the frame of the face
    cv2.imshow("Face", frameClone)
    cv2.imshow("Probabilities", canvas)

    if cv2.waitKey(1) and 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
