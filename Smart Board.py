import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start capturing video
cap = cv2.VideoCapture(0)

# Create a blank canvas to draw on
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Define colors and their positions for buttons
colors = {
    "Red": (0, 0, 255),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "Green": (0, 255, 0),
    "Eraser": (255, 255, 255)  # Not used for visual color, only for functionality
}

button_positions = {
    "Red": (540, 20),
    "Blue": (480, 20),
    "Yellow": (420, 20),
    "Green": (360, 20),
    "Eraser": (300, 20)
}

# Store previous position of the index fingertip for drawing
previous_point = None
current_color = (0, 0, 255)  # Default color is red
eraser_active = False  # To track if eraser is active

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the frame and the canvas to match each other
    frame = cv2.resize(frame, (640, 480))

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Draw buttons for color selection
    for color_name, (b, g, r) in colors.items():
        cv2.rectangle(frame, button_positions[color_name], (button_positions[color_name][0] + 40, button_positions[color_name][1] + 40), (b, g, r), -1)
        cv2.putText(frame, color_name, (button_positions[color_name][0] + 5, button_positions[color_name][1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # If at least one hand is detected
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw hand landmarks on the frame
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract wrist and all fingertips
        wrist = (int(hand_landmarks.landmark[0].x * frame.shape[1]), int(hand_landmarks.landmark[0].y * frame.shape[0]))
        index_tip = (int(hand_landmarks.landmark[8].x * frame.shape[1]), int(hand_landmarks.landmark[8].y * frame.shape[0]))
        thumb_tip = (int(hand_landmarks.landmark[4].x * frame.shape[1]), int(hand_landmarks.landmark[4].y * frame.shape[0]))

        # Detect if the index finger is not close to any other finger (for color selection)
        distance_thumb_index = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))

        if distance_thumb_index > 60:
            # Check if index finger is near any button for color selection
            for color_name, pos in button_positions.items():
                button_x, button_y = pos
                if button_x < index_tip[0] < button_x + 40 and button_y < index_tip[1] < button_y + 40:
                    if color_name == "Eraser":
                        eraser_active = True  # Activate eraser
                        canvas = np.zeros((480, 640, 3), dtype=np.uint8)  # Clear canvas
                    else:
                        current_color = colors[color_name]
                        eraser_active = False  # Deactivate eraser
                    break

        # Check if thumb and index finger are close enough to draw
        if distance_thumb_index < 40 and not eraser_active:
            if previous_point is not None:
                # Draw on the canvas (virtual board)
                cv2.line(canvas, previous_point, index_tip, current_color, 5)
            previous_point = index_tip
        else:
            previous_point = None  # Reset when thumb and index are not close enough

    else:
        # If no hands or more than one hand are detected, stop drawing
        previous_point = None

    # Combine the canvas with the frame to display both the drawing and the video feed
    combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Display the frame with drawing
    cv2.imshow("Virtual Board", combined_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()