from PIL import ImageGrab
import ctypes  # For getting screen size
import cv2
import numpy  # For Converting screenshot to a numpy array.


class ScreenReader:
    def __init__(self):
        self.is_recording = False

        # Get Screen size using ctypes
        user32 = ctypes.windll.user32
        screen_size = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        self.SCREEN_SIZE = screen_size

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.file_name = 'output.avi'
        self.out = None

    def screen_record(self):
        self.out = cv2.VideoWriter(self.file_name, self.fourcc, 20.0, self.SCREEN_SIZE)
        while True:
            img = ImageGrab.grab()
            img_np = numpy.array(img)
            frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            self.out.write(frame)

            if self.is_recording is False:
                break

        self.out.release()
        cv2.destroyAllWindows()
