import cv2
import math

# 커서 좌표와 타겟 좌표(버튼의 중심 좌표)를 받아 임계값 내면 타겟으로 스냅하는 함수
def magnet_cursor(cursor, target, threshold=20):
    dx = target[0] - cursor[0]
    dy = target[1] - cursor[1]
    distance = math.sqrt(dx**2 + dy**2)
    if distance < threshold:
        return target
    return cursor

# 예를 들어, 메인 루프 내에서
# computed_x, computed_y: MediaPipe로 계산된 손 위치에 따른 커서 좌표
# btn_x, btn_y: 버튼의 중심 좌표
computed_cursor = (computed_x, computed_y)
button_center = (btn_x, btn_y)

# 마그넷 효과 적용
final_cursor = magnet_cursor(computed_cursor, button_center, threshold=20)

# final_cursor를 사용해 화면에 커서를 그리거나 클릭 이벤트에 활용
cv2.circle(frame, final_cursor, 10, (0, 255, 0), -1)
