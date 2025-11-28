import cv2 as cv


img = cv.imread(r"C:\Users\other\codeProjects\Python\OpenCv\project 9.png")

cv.imshow("drawing", img)

def rescaleFrame(frame, scale = 0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

def changeRes(width, height):
    capture.set(3, width)
    capture.set(4, height)

resized_image = rescaleFrame(img)
cv.imshow("Image Resized", resized_image)

capture = cv.VideoCapture(0)
changeRes(1920, 1080)
print("Actual resolution:", capture.get(3), "x", capture.get(4))

while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame)

    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()


