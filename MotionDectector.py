import cv2
import time
import pandas
from datetime import datetime as dt

# Static BackGround Instance

static_back = None

# List when any moving object appears
motion_list = [None, None]

# movement time

time = []

# Initializing DataFrame, one column is start time and other column is end time
df = pandas.DataFrame(columns=["Start", "End"])

# Capture of Video

video = cv2.VideoCapture(0)

# Treat frames as video

while True:

    # Read frame from video
    check, frame = video.read()

    # Iniiate the first frame

    motion = 0

    # Convert frame to gray scale

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert gray scale to GaussianBlur so that any change can be found easily

    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if static_back is None:
        static_back = gray
        continue

    # Differentiate static background and current frame

    diff_frame = cv2.absdiff(static_back, gray)

    # Show the difference in current frame and static background if curent frame is greater than 30
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Finding contour of moving object

    conts, _ = cv2.findContours(thresh_frame.copy(),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cont in conts:
        if cv2.contourArea(cont) < 10000:
            continue
        motion = 1

        (x, y, w, h) = cv2.boundingRect(cont)

        # Create a green reactangle around the moving object

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 230, 0), 3)

    # Appending status of motion

    motion_list.append(motion)

    motion_list = motion_list[-2:]

    # Appending the first Start of motion

    if motion_list[-1] == 1 and motion_list[-2] == 0:
        time.append(dt.now())

    # Appending End of motion

    if motion_list[-1] == 0 and motion_list[-2] == 1:
        time.append(dt.now())

    # Display Image in Gray-Scale

    cv2.imshow("Gray Frame", gray)

    # Display difference in current frame and static background

    cv2.imshow("Difference Frame", diff_frame)

    # Displaying the black and White image in which if intensity value is greater than 30 it will be white else it will be black

    cv2.imshow("ThresholdFrame", thresh_frame)

    # Displaying color with contour of motion of object

    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)

    # q stops entire process

    if key == ord('q'):
        if motion == 1:
            time.append(dt.now())
        break

for i in range(0, len(time), 2):
    df = df.append({"Start": time[i],
                    "End": time[i + 1]},
                   ignore_index=True)
# Create a csv file in which start and end time of motion is stored

df.to_csv("Time.csv")

video.release()

# Destroy all windows
cv2.destroyAllWindows()
