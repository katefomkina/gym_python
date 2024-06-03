import asyncio
import websockets
import base64
import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector
import math

# Инициализация детектора позы
detector = PoseDetector(detectionCon=0.7, trackCon=0.7)

# Глобальные переменные счётчика и направления
counter = 0
direction = 0

# Класс для вычисления углов
class AngleFinder:
    def __init__(self, lmlist, p1, p2, p3, p4, p5, p6, drawPoints = True):
        self.lmlist = lmlist
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6
        self.drawPoints = drawPoints

    # Вычисление углов
    def angle(self, frame, ex_num):
        if len(self.lmlist) != 0:
            point1 = self.lmlist[self.p1]
            point2 = self.lmlist[self.p2]
            point3 = self.lmlist[self.p3]
            point4 = self.lmlist[self.p4]
            point5 = self.lmlist[self.p5]
            point6 = self.lmlist[self.p6]

            if len(point1) >= 2 and len(point2) >= 2 and len(point3) >= 2 and len(point4) >= 2 and len(
                    point5) >= 2 and len(point6) >= 2:
                x1, y1 = point1[:2]
                x2, y2 = point2[:2]
                x3, y3 = point3[:2]
                x4, y4 = point4[:2]
                x5, y5 = point5[:2]
                x6, y6 = point6[:2]

                if (ex_num == 1):
                    leftHandAngle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
                    leftHandAngle = int(np.interp(leftHandAngle, [90, 180], [100, 0]))
                    # leftHandAngle = int(np.interp(leftHandAngle, [50, 180], [100, 0]))
                elif (ex_num == 2):
                    leftHandAngle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
                    rightHandAngle = math.degrees(math.atan2(y6 - y5, x6 - x5) - math.atan2(y4 - y5, x4 - x5))
                    leftHandAngle = int(np.interp(leftHandAngle, [-30, 180], [100, 0]))
                    rightHandAngle = int(np.interp(rightHandAngle, [34, 173], [100, 0]))
                elif (ex_num == 3):
                    leftHandAngle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
                    rightHandAngle = math.degrees(math.atan2(y6 - y5, x6 - x5) - math.atan2(y4 - y5, x4 - x5))
                    leftHandAngle = int(np.interp(leftHandAngle, [-170, 180], [100, 0]))
                    rightHandAngle = int(np.interp(rightHandAngle, [-90, 40], [100, 0]))
                    # rightHandAngle = int(np.interp(rightHandAngle, [-50, 20], [100, 0]))

                # Точки и линии скелета
                if self.drawPoints:
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
                    cv2.line(frame, (x2, y2), (x3, y3), (255, 255, 255), 3)
                    cv2.line(frame, (x4, y4), (x5, y5), (255, 255, 255), 3)
                    cv2.line(frame, (x5, y5), (x6, y6), (255, 255, 255), 3)
                    cv2.line(frame, (x1, y1), (x4, y4), (255, 255, 255), 3)

                    cv2.circle(frame, (x1, y1), 1, (255, 255, 255), 14)
                    cv2.circle(frame, (x2, y2), 1, (255, 255, 255), 14)
                    cv2.circle(frame, (x3, y3), 1, (255, 255, 255), 14)
                    cv2.circle(frame, (x4, y4), 1, (255, 255, 255), 14)
                    cv2.circle(frame, (x5, y5), 1, (255, 255, 255), 14)
                    cv2.circle(frame, (x6, y6), 1, (255, 255, 255), 14)

                if (ex_num == 1):
                    return leftHandAngle
                if (ex_num == 2 or ex_num == 3):
                    return frame, [leftHandAngle, rightHandAngle]


async def process_video(websocket):
    global counter, direction
    frame_counter = 0  # Счётчик кадров

    async for message in websocket:
        frame_counter += 1  # Увеличиваем счётчик каждый раз, когда приходит кадр

        # Пропускаем обработку, если кадр не каждый (n)
        if frame_counter % 10 != 0:
            continue

        # Декодирование изображения из base64
        #img_data = base64.b64decode(message.split(',')[1])
        img_and_num = message.split(',')
        img_data = base64.b64decode(img_and_num[1].encode())
        ex_num = int(img_and_num[2])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        try:
            # Обработка кадра
            detector.findPose(frame, draw=0)
            lmList, bboxInfo = detector.findPosition(frame, bboxWithHands=0, draw=False)

            if (ex_num == 1):
                angle1 = AngleFinder(lmList, 24, 26, 28, 23, 25, 27, drawPoints=True)
                left = angle1.angle(frame, ex_num)
                if left >= 90:
                    if direction == 0:
                        counter += 0.5
                        direction = 1
                if left <= 70:
                    if direction == 1:
                        counter += 0.5
                        direction = 0
            elif (ex_num == 2):
                angle1 = AngleFinder(lmList, 11, 13, 15, 12, 14, 16, drawPoints=True)
                frame, hands = angle1.angle(frame, ex_num)
                if not hands:
                    continue
                left, right = hands[0:]
                if left >= 70 and right >= 70:
                    if direction == 0:
                        counter += 0.5
                        direction = 1
                if left <= 70 and right <= 70:
                    if direction == 1:
                        counter += 0.5
                        direction = 0
            elif (ex_num == 3):
                angle1 = AngleFinder(lmList, 11, 13, 15, 12, 14, 16, drawPoints=True)
                frame, hands = angle1.angle(frame, ex_num)
                if not hands:
                    continue
                left, right = hands[0:]
                if left >= 90 and right >= 90:
                    if direction == 0:
                        counter += 0.5
                        direction = 1
                if left <= 70 and right <= 70:
                    if direction == 1:
                        counter += 0.5
                        direction = 0

        # Если человек вышел за пределы кадра
        except Exception as e:
            print(f"An error occurred: {e}")

        # Кодирование обработанного кадра обратно в base64
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_frame = base64.b64encode(buffer).decode('utf-8')

        message = f"{encoded_frame},{str(int(counter))}"

        # Отправка сообщения клиенту
        await websocket.send(message)

start_server = websockets.serve(process_video, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()