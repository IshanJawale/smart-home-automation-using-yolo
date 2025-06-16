import cv2

# Function for motion tracking
def track_motion(prev_frame, frame, boxes):
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow
    flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    for box in boxes:
        x, y, w, h = box
        center_x = x + w // 2
        center_y = y + h // 2

        # Get flow vector for the center point of the bounding box
        flow_vector = flow[center_y, center_x]
        end_point = (int(center_x + flow_vector[0]), int(center_y + flow_vector[1]))

        # Draw the motion path
        cv2.arrowedLine(frame, (center_x, center_y), end_point, (0, 0, 255), 2)
