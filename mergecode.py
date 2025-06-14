import cv2
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen

# ---------- OOP ตรวจจับสี่เหลี่ยม ----------
class BrownRectangleDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original = None
        self.image = None
        self.hsv = None
        self.mask = None
        self.contours = []
        self.detected_image = None

    def load_image(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        self.original = self.image.copy()
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    def create_mask(self):
        lower_brown = np.array([10, 100, 20])
        upper_brown = np.array([35, 255, 200])
        self.mask = cv2.inRange(self.hsv, lower_brown, upper_brown)

        kernel = np.ones((5, 5), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)

    def find_contours(self):
        self.contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def process_contours(self):
        for cnt in self.contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area > 500:
                    cv2.drawContours(self.image, [approx], -1, (0, 255, 0), 3)

        self.detected_image = self.image.copy()

    def run(self):
        self.load_image()
        self.create_mask()
        self.find_contours()
        self.process_contours()
        return self.original, self.detected_image, self.mask

# ---------- Screens ----------
class MainScreen(Screen):
    pass

class ResultScreen(Screen):
    pass

# ---------- Kivy UI ----------
class CamApp(App):
    def build(self):
        self.sm = ScreenManager()

        self.main_screen = MainScreen(name='main')
        self.result_screen = ResultScreen(name='result')

        self.build_main_screen()
        self.build_result_screen()

        self.sm.add_widget(self.main_screen)
        self.sm.add_widget(self.result_screen)

        return self.sm

    def build_main_screen(self):
        layout = BoxLayout(orientation='horizontal')
        self.img_widget = Image()
        left = BoxLayout(size_hint=(0.7, 1))
        left.add_widget(self.img_widget)

        right = BoxLayout(orientation='vertical', spacing=20, padding=40, size_hint=(0.3, 1))
        self.btn = Button(text='open camera', size_hint=(1, None), height=70)
        self.btn.bind(on_press=self.toggle_camera)

        capture_btn = Button(text='capture', size_hint=(1, None), height=70)
        capture_btn.bind(on_press=self.capture_image)

        detect_btn = Button(text='detect', size_hint=(1, None), height=70)
        detect_btn.bind(on_press=self.detect)

        exit_btn = Button(text='exit', size_hint=(1, None), height=70)
        exit_btn.bind(on_press=self.stop_app)

        for btn in [self.btn, capture_btn, detect_btn, exit_btn]:
            right.add_widget(btn)

        layout.add_widget(left)
        layout.add_widget(right)
        self.main_screen.add_widget(layout)

    def build_result_screen(self):
        layout = BoxLayout(orientation='horizontal')
        self.detected_img_widget = Image()
        self.mask_img_widget = Image()

        left = BoxLayout(size_hint=(0.7, 1))
        left.add_widget(self.detected_img_widget)
        left.add_widget(self.mask_img_widget)

        right = BoxLayout(orientation='vertical', spacing=20, padding=40, size_hint=(0.3, 1))
        back_btn = Button(text='back', size_hint=(1, None), height=70)
        back_btn.bind(on_press=self.go_back)

        export_btn = Button(text='export', size_hint=(1, None), height=70)
        export_btn.bind(on_press=self.export)

        exit_btn = Button(text='exit', size_hint=(1, None), height=70)
        exit_btn.bind(on_press=self.stop_app)

        for btn in [back_btn, export_btn, exit_btn]:
            right.add_widget(btn)

        layout.add_widget(left)
        layout.add_widget(right)
        self.result_screen.add_widget(layout)

    def toggle_camera(self, instance):
        if hasattr(self, 'capture') and self.capture:
            self.capture.release()
            self.capture = None
            self.img_widget.texture = None
            self.btn.text = 'open camera'
            Clock.unschedule(self.update)
        else:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                print("Cannot open camera")
                return
            self.btn.text = 'close camera'
            Clock.schedule_interval(self.update, 1.0 / 60.0)

    def update(self, dt):
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.flip(frame_rgb, 0)
                buf = frame_rgb.tobytes()
                texture = Texture.create(size=(frame_rgb.shape[1], frame_rgb.shape[0]), colorfmt='rgb')
                texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
                self.img_widget.texture = texture

    def capture_image(self, instance):
        if hasattr(self, 'capture') and self.capture:
            ret, frame = self.capture.read()
            if ret:
                cv2.imwrite("captured_image.jpg", frame)
                print("Image captured.")

    def detect(self, instance):
        try:
            detector = BrownRectangleDetector("captured_image.jpg")
            original, detected, mask = detector.run()

            # detected image
            detected_rgb = cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)
            detected_rgb = cv2.flip(detected_rgb, 0)
            buf1 = detected_rgb.tobytes()
            texture1 = Texture.create(size=(detected_rgb.shape[1], detected_rgb.shape[0]), colorfmt='rgb')
            texture1.blit_buffer(buf1, colorfmt='rgb', bufferfmt='ubyte')
            self.detected_img_widget.texture = texture1

            # mask image
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_rgb = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)
            mask_rgb = cv2.flip(mask_rgb, 0)
            buf2 = mask_rgb.tobytes()
            texture2 = Texture.create(size=(mask_rgb.shape[1], mask_rgb.shape[0]), colorfmt='rgb')
            texture2.blit_buffer(buf2, colorfmt='rgb', bufferfmt='ubyte')
            self.mask_img_widget.texture = texture2

            self.sm.current = 'result'

        except Exception as e:
            print(f"Detection error: {e}")

    def go_back(self, instance):
        self.sm.current = 'main'

    def export(self, instance):
        print("Export function placeholder")

    def stop_app(self, instance):
        App.get_running_app().stop()
        if hasattr(self, 'capture') and self.capture:
            self.capture.release()
        Window.close()

if __name__ == '__main__':
    CamApp().run()
