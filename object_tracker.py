import numpy as np
import cv2


class TrackObject:
    def __init__(self, video_file):
        self.video_file = video_file
        self.coords = (0, 0, 0, 0)
        self.width, self.height = self.get_property()

    def get_property(self):
        cap = cv2.VideoCapture(self.video_file)
        if cap.isOpened():
            width = int(cap.get(3))
            height = int(cap.get(4))
        return width, height

    def tracking(self):
        cap = cv2.VideoCapture(self.video_file)
        # Read the first frame of the video
        ret, frame = cap.read()
        # Set the ROI
        c, r, w, h = 400, 580, 100, 150  # for apple.mp4
        track_window = (c, r, w, h)
        # Create mask and normalized histogram
        roi = frame[r:r + h, c:c + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 30., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1)

        while True:
            ret, frame = cap.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            x, y, w, h = track_window
            cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
            cv2.putText(frame, 'Tracking', (x - 25, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            self.coords = (x, y, x + w, y + h)
            print('top-left centered', self.coords)
            # need to reset so camera center as (0,0)
            self.coords = (x - self.width/2,
                           y - self.height/2,
                           x+w-self.width/2,
                           y+h-self.height/2)
            print('camera centroid centered', self.coords)
            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    track_object = TrackObject('apple.mp4')
    track_object.tracking()
    track_window_coords = track_object.coords
    print(track_window_coords)
