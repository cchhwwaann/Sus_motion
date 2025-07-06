import cv2
import dlib
import numpy as np
from pynput.mouse import Controller
import time
import os 
import screeninfo

# --- Configuration Settings ---
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat" # Path to the downloaded dlib model file
BLINK_THRESHOLD = 0.25 # Eye Aspect Ratio (EAR) threshold to determine if eyes are closed
BLINK_COUNT_INTERVAL_MS = 1000 # Time window (milliseconds) to count consecutive blinks
BLINK_REQUIRED_FOR_TOGGLE = 3 # Number of consecutive blinks required to toggle control mode
EYE_MOVE_SENSITIVITY = 0.8 # Sensitivity of mouse cursor movement based on eye movement (adjustable, recommended 0.5 to 2.0)
SMOOTHING_FACTOR = 0.2 # Degree of smoothness for mouse cursor movement (0.01 to 1.0, lower for smoother)

# --- Global Variables ---
mouse = Controller()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

blink_count = 0
last_blink_time = time.time()
control_mode_active = False # Flag to indicate if GUI control mode is active

# Calibration related variables
calibration_points = [] # Stores eye relative positions (x, y) collected during calibration
calibration_screen_coords = [ # Normalized target screen coordinates for 9-point calibration
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1), # Top-left, Top-center, Top-right
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5), # Mid-left, Center, Mid-right
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)  # Bottom-left, Bottom-center, Bottom-right
]
calibration_step = 0 # Current step in the calibration process (starts from 0)
is_calibrating = False # Flag to indicate if calibration mode is active
calibration_data_collected = False # Flag to indicate if calibration is complete

# Variables to store the calibrated eye movement range
# These will be updated after calibration is complete
calibrated_min_eye_x, calibrated_max_eye_x = 0.0, 1.0
calibrated_min_eye_y, calibrated_max_eye_y = 0.0, 1.0

# --- Utility Functions ---

def euclidean_dist(ptA, ptB):
    """Calculates the Euclidean distance between two points."""
    return np.linalg.norm(np.array(ptA) - np.array(ptB))

def eye_aspect_ratio(eye):
    """
    Calculates the Eye Aspect Ratio (EAR) to determine if the eye is open or closed.
    Takes eye landmark coordinates as input.
    """
    # Vertical distances between eye landmarks
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    # Horizontal distance between eye landmarks
    C = euclidean_dist(eye[0], eye[3])
    # EAR calculation
    ear = (A + B) / (2.0 * C)
    return ear

def get_landmarks(image, detector, predictor):
    """
    Detects face landmarks in an image and returns the landmarks and the face rectangle.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0) # Detect faces
    
    if len(rects) > 0: # If a face is detected
        rect = rects[0] # Use the first detected face
        shape = predictor(gray, rect)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        return landmarks, rect # Return both landmarks and the rectangle object
    
    return None, None # If no face is found, return None for both


def get_eye_center(landmarks, eye_indices):
    """
    Calculates the center coordinates of the eye using the given landmark indices.
    """
    eye_pts = landmarks[eye_indices]
    eye_center = np.mean(eye_pts, axis=0).astype(int)
    return eye_center

# --- Main Loop ---
cap = cv2.VideoCapture(0) # Initialize webcam (0 is default webcam)
if not cap.isOpened():
    print("Failed to open webcam.")
    exit()

# Get primary monitor information using screeninfo
screen = screeninfo.get_monitors()[0]
screen_width = screen.width
screen_height = screen.height
print(f"Screen Resolution: {screen_width}x{screen_height}")

print("--- Eye Tracking GUI Control Program Started ---")
print("To start calibration: Press 'c' key (9 steps total)")
print(f"To toggle mouse control mode ON/OFF: Blink your eyes {BLINK_REQUIRED_FOR_TOGGLE} times consecutively.")
print(f"Current Control Mode: {'Active' if control_mode_active else 'Inactive'}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # Flip frame horizontally (mirror mode)
    
    # If calibrating, dim the background
    if is_calibrating:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        alpha = 0.7 # Transparency
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Get landmarks and face rectangle
    landmarks, rect = get_landmarks(frame, detector, predictor)
    
    # Proceed only if landmarks and face rectangle are valid
    if landmarks is not None and rect is not None:
        # Eye landmark indices (based on dlib 68 landmarks)
        # Left eye: (36, 37, 38, 39, 40, 41)
        # Right eye: (42, 43, 44, 45, 46, 47)
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        # Calculate eye centers
        left_eye_center = get_eye_center(landmarks, [37, 38, 40, 41]) # Using landmarks around the pupil
        right_eye_center = get_eye_center(landmarks, [43, 44, 46, 47])

        # Eye blink detection (using EAR)
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0 # Average EAR of both eyes

        # Eye blink count logic (blink toggle is inactive during calibration)
        if not is_calibrating:
            if ear < BLINK_THRESHOLD: # If eye is closed
                current_time = time.time()
                if current_time - last_blink_time > 0.3: # Prevent duplicate blinks (0.3 sec cooldown)
                    blink_count += 1
                    last_blink_time = current_time
                    print(f"Blink detected! Current blink count: {blink_count}")
            elif blink_count > 0 and (time.time() - last_blink_time) * 1000 > BLINK_COUNT_INTERVAL_MS:
                # Reset count if no additional blinks within interval
                if blink_count >= BLINK_REQUIRED_FOR_TOGGLE:
                    control_mode_active = not control_mode_active
                    print(f"Control Mode Toggled! Current Control Mode: {'Active' if control_mode_active else 'Inactive'}")
                blink_count = 0
            
        # Calculate eye position relative to face bounding box
        eye_pos = (left_eye_center + right_eye_center) / 2
        (x_face, y_face, w_face, h_face) = (rect.left(), rect.top(), rect.width(), rect.height())
        eye_pos_relative_x = (eye_pos[0] - x_face) / w_face
        eye_pos_relative_y = (eye_pos[1] - y_face) / h_face

        # Calibration Mode Logic
        if is_calibrating:
            if calibration_step < len(calibration_screen_coords):
                # Draw calibration point
                target_norm_x, target_norm_y = calibration_screen_coords[calibration_step]
                target_pixel_x = int(target_norm_x * frame.shape[1])
                target_pixel_y = int(target_norm_y * frame.shape[0])
                cv2.circle(frame, (target_pixel_x, target_pixel_y), 15, (0, 255, 255), -1) # Yellow circle
                cv2.putText(frame, f"Look at the dot ({calibration_step + 1}/{len(calibration_screen_coords)})",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Press SPACE to record", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            else: # All calibration steps completed
                is_calibrating = False
                calibration_data_collected = True
                print("\n--- Calibration Complete! ---")

                # Calculate the eye movement range from collected calibration data
                if calibration_points:
                    eye_x_coords = [p[0] for p in calibration_points]
                    eye_y_coords = [p[1] for p in calibration_points]
                    
                    # Add a small buffer to min/max to avoid division by zero or out-of-bounds issues
                    buffer_x = 0.05 * (max(eye_x_coords) - min(eye_x_coords)) if (max(eye_x_coords) - min(eye_x_coords)) > 0 else 0.01
                    buffer_y = 0.05 * (max(eye_y_coords) - min(eye_y_coords)) if (max(eye_y_coords) - min(eye_y_coords)) > 0 else 0.01

                    calibrated_min_eye_x = min(eye_x_coords) - buffer_x
                    calibrated_max_eye_x = max(eye_x_coords) + buffer_x
                    calibrated_min_eye_y = min(eye_y_coords) - buffer_y
                    calibrated_max_eye_y = max(eye_y_coords) + buffer_y

                    print(f"Calibrated Eye X Range: [{calibrated_min_eye_x:.2f}, {calibrated_max_eye_x:.2f}]")
                    print(f"Calibrated Eye Y Range: [{calibrated_min_eye_y:.2f}, {calibrated_max_eye_y:.2f}]")
                else:
                    print("No calibration data collected. Using default ranges.")
                    # If no data, default ranges (0.0-1.0) will be used, which might not be accurate.

        # GUI Control Mode (active only if control_mode_active is True AND calibration is complete)
        if control_mode_active and calibration_data_collected:
            # Normalize eye relative position (0-1 range) based on calibrated range
            # Add a small check to prevent division by zero if range is too small
            if (calibrated_max_eye_x - calibrated_min_eye_x) > 0.001: # Minimum valid range
                mapped_x_norm = (eye_pos_relative_x - calibrated_min_eye_x) / (calibrated_max_eye_x - calibrated_min_eye_x)
            else:
                mapped_x_norm = 0.5 # Default to center if range is invalid
            
            if (calibrated_max_eye_y - calibrated_min_eye_y) > 0.001: # Minimum valid range
                mapped_y_norm = (eye_pos_relative_y - calibrated_min_eye_y) / (calibrated_max_eye_y - calibrated_min_eye_y)
            else:
                mapped_y_norm = 0.5 # Default to center if range is invalid

            # Clamp mapped values to ensure they stay within 0-1 range
            mapped_x_norm = max(0.0, min(1.0, mapped_x_norm))
            mapped_y_norm = max(0.0, min(1.0, mapped_y_norm))

            # Calculate final mouse target coordinates based on screen resolution
            target_x = mapped_x_norm * screen_width
            target_y = mapped_y_norm * screen_height

            # Smooth mouse movement using SMOOTHING_FACTOR
            current_mouse_x, current_mouse_y = mouse.position

            smoothed_x = current_mouse_x + (target_x - current_mouse_x) * SMOOTHING_FACTOR
            smoothed_y = current_mouse_y + (target_y - current_mouse_y) * SMOOTHING_FACTOR

            # Clamp mouse position to screen boundaries
            smoothed_x = max(0, min(screen_width - 1, smoothed_x))
            smoothed_y = max(0, min(screen_height - 1, smoothed_y))

            mouse.position = (int(smoothed_x), int(smoothed_y)) # Move mouse

            # Visualize eye position (green dot)
            cv2.circle(frame, tuple(eye_pos.astype(int)), 5, (0, 255, 0), -1)

        # Visualize face landmarks
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        
        # Display EAR value
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Control Mode: {'ON' if control_mode_active else 'OFF'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display calibration status
        if is_calibrating:
            cv2.putText(frame, "CALIBRATING...", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif not calibration_data_collected:
             cv2.putText(frame, "Press 'c' to Calibrate", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


    cv2.imshow("Eye Tracking for GUI Control", frame)

    # Key press handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): # Press 'q' to quit
        break
    elif key == ord('c') and not is_calibrating: # Press 'c' to start calibration
        is_calibrating = True
        calibration_step = 0
        calibration_points = [] # Clear existing data
        calibration_data_collected = False
        print("\n--- Starting Calibration ---")
        print("Look at the yellow dot on the screen and press SPACEBAR to record each point.")
    elif key == ord(' ') and is_calibrating: # Press spacebar during calibration to record point
        # Ensure landmarks and rect are available before recording
        if landmarks is not None and rect is not None:
            if calibration_step < len(calibration_screen_coords):
                calibration_points.append((eye_pos_relative_x, eye_pos_relative_y))
                calibration_step += 1
                print(f"Calibration point {calibration_step} collected.")
            else:
                # This else block should ideally not be reached if calibration_step is managed correctly
                # but as a fallback, if spacebar is pressed after all steps, ensure calibration ends.
                is_calibrating = False
                calibration_data_collected = True
                print("Calibration complete via extra spacebar press.")
        else:
            print("Cannot record calibration point: Face/eyes not detected. Please ensure your face is visible.")


cap.release()
cv2.destroyAllWindows()
print("--- Program Exited ---")
### 폐기..! ㅠ