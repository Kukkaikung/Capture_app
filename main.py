import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.core.window import Window

class CamApp(App):
    def build(self):
        self.capture = None
        self.is_running = False
        main_layout = BoxLayout(orientation='horizontal')

        # ซ้าย
        self.img_widget = Image()
        left_layout = BoxLayout(size_hint=(0.7, 1))
        left_layout.add_widget(self.img_widget)

        # ขวา
        right_layout = BoxLayout(orientation='vertical', spacing=50, padding=40, size_hint=(0.3, 1))

        # ปุ่ม1
        self.btn = Button(text='open camera', size_hint=(1, None), height=70)
        self.btn.bind(on_press=self.toggle_camera)

        # ปุ่ม2
        btn2 = Button(text='capture', size_hint=(1, None), height=70)
        btn2.bind(on_press=self.capture_image)

        # ปุ่ม3
        btn3 = Button(text='exit', size_hint=(1, None), height=70)
        btn3.bind(on_press=self.stop_app)

        # ปุ่ม3
        btn4 = Button(text='detect', size_hint=(1, None), height=70)
        btn4.bind(on_press=self.detect)
        # เพิ่มปุ่มในขวา
        right_layout.add_widget(self.btn)
        right_layout.add_widget(btn2)
        right_layout.add_widget(btn4)
        right_layout.add_widget(btn3)

        # รวม
        main_layout.add_widget(left_layout)
        main_layout.add_widget(right_layout)

        return main_layout

    def toggle_camera(self, instance):
        if self.is_running:
            self.is_running = False
            self.btn.text = "open camera"
            if self.capture:
                self.capture.release()
                self.capture = None
            Clock.unschedule(self.update)
            self.img_widget.texture = None
        else:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                print("Error: cannot open camera")
                return
            self.is_running = True
            self.btn.text = "close camera"
            Clock.schedule_interval(self.update, 1.0 / 60.0)

    def update(self, dt):
        if self.capture and self.is_running:
            ret, frame = self.capture.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 0)  
                buf = frame.tobytes()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
                texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
                self.img_widget.texture = texture
    def detect(self, instance):
        return
    def capture_image(self, instance):
        if self.capture and self.is_running:
            ret, frame = self.capture.read()
            if ret:
                cv2.imwrite("captured_image.jpg", frame)

    def stop_app(self, instance):
        App.get_running_app().stop()
        if self.capture:
            self.capture.release()
        Window.close()

if __name__ == '__main__':
    CamApp().run()
