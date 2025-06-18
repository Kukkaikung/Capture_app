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

# ---------- OOP ตรวจจับและแปลง Perspective ----------
class BrownRectangleDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original = None
        self.image = None
        self.hsv = None
        self.mask = None
        self.contours = []
        self.warped = None

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
                    pts = approx.reshape(4, 2)
                    rect = self.order_points(pts)
                    self.warp_perspective(rect)
                    break  # ใช้เฉพาะสี่เหลี่ยมแรก

    def warp_perspective(self, rect):
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        self.warped = cv2.warpPerspective(self.original, M, (maxWidth, maxHeight))

    def run(self):
        self.load_image()
        self.create_mask()
        self.find_contours()
        self.process_contours()
        return self.original, self.warped

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

        self.original_img_widget = Image()
        self.warped_img_widget = Image()

        left = BoxLayout(size_hint=(0.5, 1))
        left.add_widget(self.original_img_widget)

        right = BoxLayout(size_hint=(0.5, 1))
        right.add_widget(self.warped_img_widget)

        btns = BoxLayout(orientation='vertical', spacing=20, padding=40, size_hint=(0.3, 1))
        back_btn = Button(text='back', size_hint=(1, None), height=70)
        back_btn.bind(on_press=self.go_back)

        export_btn = Button(text='export', size_hint=(1, None), height=70)
        export_btn.bind(on_press=self.export)

        exit_btn = Button(text='exit', size_hint=(1, None), height=70)
        exit_btn.bind(on_press=self.stop_app)

        for btn in [back_btn, export_btn, exit_btn]:
            btns.add_widget(btn)

        layout.add_widget(left)
        layout.add_widget(right)
        layout.add_widget(btns)

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
            original, warped = detector.run()

            if original is not None:
                original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                original_rgb = cv2.flip(original_rgb, 0)
                buf_orig = original_rgb.tobytes()
                texture_orig = Texture.create(size=(original_rgb.shape[1], original_rgb.shape[0]), colorfmt='rgb')
                texture_orig.blit_buffer(buf_orig, colorfmt='rgb', bufferfmt='ubyte')
                self.original_img_widget.texture = texture_orig

            if warped is not None:
                warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                warped_rgb = cv2.flip(warped_rgb, 0)
                buf_warp = warped_rgb.tobytes()
                texture_warp = Texture.create(size=(warped_rgb.shape[1], warped_rgb.shape[0]), colorfmt='rgb')
                texture_warp.blit_buffer(buf_warp, colorfmt='rgb', bufferfmt='ubyte')
                self.warped_img_widget.texture = texture_warp

            self.sm.current = 'result'

        except Exception as e:
            print(f"Detection error: {e}")

    def go_back(self, instance):
        self.sm.current = 'main'

    def export(self, instance):
        if hasattr(self, 'warped_img_widget') and self.warped_img_widget.texture:
            try:
                # ดึงข้อมูลจาก texture ของ warped image
                size = self.warped_img_widget.texture.size
                pixels = self.warped_img_widget.texture.pixels
                img = np.frombuffer(pixels, dtype=np.uint8).reshape(size[1], size[0], 4)

                # ลบ alpha channel (RGBA -> RGB)
                img = img[:, :, :3]

                # Flip กลับ (เนื่องจากเรากลับแกน y ตอนแสดงภาพ)
                img = cv2.flip(img, 0)

                # แปลงจาก RGB กลับเป็น BGR เพื่อให้ใช้กับ OpenCV
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # เซฟภาพ
                cv2.imwrite("exported_warped.jpg", img_bgr)
                print("Warped image exported as 'exported_warped.jpg'")
            except Exception as e:
                print(f"Error exporting image: {e}")
        else:
            print("No warped image to export.")


    def stop_app(self, instance):
        App.get_running_app().stop()
        if hasattr(self, 'capture') and self.capture:
            self.capture.release()
        Window.close()

if __name__ == '__main__':
    CamApp().run()
