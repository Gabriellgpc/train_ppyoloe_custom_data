import cv2

def main():
    video = cv2.VideoCapture('/dev/video0')

    while True:
        ret, frame = video.read()

        if ret == False:
            break

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__=='__main__':
    main()