from PIL import ImageGrab
import cv2
import platform
import numpy  # For Converting screenshot to a numpy array.

PLATFORM = platform.system()


class ScreenReader:
    def __init__(self):
        self.is_recording = False

        if PLATFORM == "Windows":
            import ctypes  # For getting screen size

            # Get Screen size using ctypes for windows
            user32 = ctypes.windll.user32
            screen_size = user32.GetSystemMetrics(
                0), user32.GetSystemMetrics(1)
        # elif PLATFORM == "Darwin":
        #     from AppKit import NSScreen
        #
        #     screen_size = (
        #         int(NSScreen.mainScreen().frame().size.width),
        #         int(NSScreen.mainScreen().frame().size.height),
        #     )
        self.SCREEN_SIZE = screen_size
        print(self.SCREEN_SIZE)

        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.file_name = "output.avi"
        self.out = None

    def screen_record(self):
        self.out = cv2.VideoWriter(filename=self.file_name, fourcc=self.fourcc, fps=20.0, frameSize=self.SCREEN_SIZE)

        # self.out = cv2.VideoWriter(
            # filename=self.file_name, apiPreference=None, params=None, fourcc=self.fourcc, fps=20, frameSize=self.SCREEN_SIZE)

        while True:
            img = ImageGrab.grab()
            img_np = numpy.array(img)
            frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            self.out.write(frame)

            if self.is_recording is False:
                break

        self.out.release()
        cv2.destroyAllWindows()
