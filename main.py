import cv2

# Loading pre-trained classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Capturing video stream from webcam
cap = cv2.VideoCapture(0)

# Creating window for video output
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

# Colors for displaying different elements
color_green = (0, 255, 0)
color_blue = (0, 0, 255)

while True:
    # Reading frame
    ret, frame = cap.read()

    if not ret:
        break

    # Converting to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Drawing rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_green, 2)

        # Detecting eyes within the face area
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, minNeighbors=5, minSize=(30, 30))

        # Limiting the number of detected eyes to two
        eyes = eyes[:2]

        # Checking if eyes are on the same horizontal line
        if len(eyes) == 2:
            (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes
            if abs((ey1 + eh1 // 2) - (ey2 + eh2 // 2)) < eh1 // 2:
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), color_blue, 2)

                # Computing coordinates of eye centers
                left_eye_center = (x + ex1 + ew1 // 2, y + ey1 + eh1 // 2)
                right_eye_center = (x + ex2 + ew2 // 2, y + ey2 + eh2 // 2)

                # Drawing a horizontal line from the center of the left eye to the center of the right eye
                cv2.line(frame, left_eye_center, right_eye_center, color_blue, 2)

    # Displaying the frame in the window
    cv2.imshow('Video', frame)

    # Exiting on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing resources
cap.release()
cv2.destroyAllWindows()
