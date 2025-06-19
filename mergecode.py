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
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.filechooser import FileChooserListView
import os
from datetime import datetime

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
                    break

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
        self.capture_count = 0  
        self.save_folder = "images"
        os.makedirs(self.save_folder, exist_ok=True)
        self.capture_count = self.sort() 
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

        select_cam_btn = Button(text='select camera', size_hint=(1, None), height=70)
        select_cam_btn.bind(on_press=self.select_camera)

        self.btn = Button(text='open camera', size_hint=(1, None), height=70)
        self.btn.bind(on_press=self.toggle_camera)

        capture_btn = Button(text='capture', size_hint=(1, None), height=70)
        capture_btn.bind(on_press=self.capture_image)

        detect_btn = Button(text='detect', size_hint=(1, None), height=70)
        detect_btn.bind(on_press=self.select_image_and_detect)

        exit_btn = Button(text='exit', size_hint=(1, None), height=70)
        exit_btn.bind(on_press=self.stop_app)

        for btn in [select_cam_btn, self.btn, capture_btn, detect_btn, exit_btn]:
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

    def select_camera(self, instance):
        content = BoxLayout(orientation='vertical', spacing=10, padding=20)

        label = Label(text="Enter Camera Index (0, 1, 2...):", size_hint=(1, None), height=30)
        cam_input = TextInput(multiline=False, size_hint=(1, None), height=50, input_filter='int')

        btn_ok = Button(text="Set", size_hint=(1, None), height=50)
        btn_cancel = Button(text="Cancel", size_hint=(1, None), height=50)

        content.add_widget(label)
        content.add_widget(cam_input)
        content.add_widget(btn_ok)
        content.add_widget(btn_cancel)

        popup = Popup(title="Select Camera", content=content,
                      size_hint=(None, None), size=(500,500), auto_dismiss=False)

        def set_camera(instance):
            try:
                idx = int(cam_input.text)
                self.camera_index = idx
                print(f"Camera index set to {idx}")
            except ValueError:
                print("Invalid camera index.")
            popup.dismiss()

        def cancel(instance):
            popup.dismiss()

        btn_ok.bind(on_press=set_camera)
        btn_cancel.bind(on_press=cancel)

        popup.open()

    def toggle_camera(self, instance):
        if hasattr(self, 'capture') and self.capture:
            self.capture.release()
            self.capture = None
            self.img_widget.texture = None
            self.btn.text = 'open camera'
            Clock.unschedule(self.update)
        else:
            self.capture = cv2.VideoCapture(self.camera_index)
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
    def detect_image_file(self, image_path):
        try:
            detector = BrownRectangleDetector(image_path)
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

    def select_image_and_detect(self, instance):
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        filechooser = FileChooserListView(path='images/', filters=['*.jpg'])
        btn_box = BoxLayout(size_hint=(1, None), height=50, spacing=10)
        btn_ok = Button(text='Detect')
        btn_cancel = Button(text='Cancel')

        btn_box.add_widget(btn_ok)
        btn_box.add_widget(btn_cancel)

        content.add_widget(filechooser)
        content.add_widget(btn_box)

        popup = Popup(title='Select image to detect', content=content,
                    size_hint=(0.9, 0.9), auto_dismiss=False)

        def detect_selected_image(instance):
            if filechooser.selection:
                selected_path = filechooser.selection[0]
                popup.dismiss()
                self.detect_image_file(selected_path)
            else:
                print("No file selected.")

        def cancel_popup(instance):
            popup.dismiss()

        btn_ok.bind(on_press=detect_selected_image)
        btn_cancel.bind(on_press=cancel_popup)

        popup.open()

    def capture_image(self, instance):
        if hasattr(self, 'capture') and self.capture:
            ret, frame = self.capture.read()
             
            if ret:
                i = 1
            self.capture_count += 1
            filename = os.path.join(self.save_folder,f"captured_image{self.capture_count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Image captured and saved as '{filename}'")
               
    def sort(self):
        files = os.listdir(self.save_folder)
        max_number = 0
        for file in files:
            if file.startswith("captured_image") and file.endswith(".jpg"):
                try:
                    number = int(file.replace("captured_image", "").replace(".jpg", ""))
                    if number > max_number:
                        max_number = number
                except:
                    continue
        return max_number

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
        def save_warped(instance):
            try:
                if self.warped_img_widget.texture:
                    now = datetime.now().strftime('%Y%m%d_%H%M%S')
            
                    export_folder = os.path.join(self.save_folder, "detected_img")
                    os.makedirs(export_folder, exist_ok=True)
            
                    filename = os.path.join(export_folder, f"warped_{now}.jpg")

                    size = self.warped_img_widget.texture.size
                    pixels = self.warped_img_widget.texture.pixels
                    img = np.frombuffer(pixels, dtype=np.uint8).reshape(size[1], size[0], 4)
                    img = img[:, :, :3]
                    img = cv2.flip(img, 0)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filename, img_bgr)
                    print(f"Warped image exported as '{filename}'")
            except Exception as e:
                print(f"Error exporting warped image: {e}")
            popup.dismiss()


        def close_popup(instance):
            popup.dismiss()

    # --- ส่วนนี้สำคัญ: ต้องอยู่ก่อน popup.open() ---
        content = BoxLayout(orientation='vertical', spacing=10, padding=20)

        label = Label(text="Export Warped Image", size_hint=(1, None), height=30)
        btn_export = Button(text="Export", size_hint=(1, None), height=50)
        btn_cancel = Button(text="Cancel", size_hint=(1, None), height=50)

        content.add_widget(label)
        content.add_widget(btn_export)
        content.add_widget(btn_cancel)

        popup = Popup(title="Export Warped Image", content=content,
                    size_hint=(None, None), size=(400, 250), auto_dismiss=False)

        btn_export.bind(on_press=save_warped)
        btn_cancel.bind(on_press=close_popup)

        popup.open()

    def stop_app(self, instance):
        App.get_running_app().stop()
        if hasattr(self, 'capture') and self.capture:
            self.capture.release()
        Window.close()

if __name__ == '__main__':
    CamApp().run()
