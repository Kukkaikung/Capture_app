################################
# กรณีไฟลอยู่โฟลเดอรเดียวกัน
from BrownRec import BrownRectangleDetector

if __name__ == "__main__":
    image_path = r"C:\Users\ASUS\Desktop\Pill classification\perspective tranform\pic\pic09.jpg"
    detector = BrownRectangleDetector(image_path)
    detector.run()
################################


#กรณีไฟลอยู่คนล่ะโฟลเดอร
# import sys
# sys.path.append('./test_oop')

# from only_oop import BrownRectangleDetector

# if __name__ == "__main__":
#     image_path = r"C:\Users\ASUS\Desktop\Pill classification\perspective tranform\pic\pic09.jpg"
#     detector = BrownRectangleDetector(image_path)
#     detector.run()
