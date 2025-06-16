import cv2
from yolo_model import load_yolo
from people_detection import detect_people
from motion_tracking import track_motion

def detect_motion_and_people(classes):
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("Error: Could not open webcam.")
        return 0  # Return 0 if webcam cannot be opened

    net, output_layers = load_yolo()
    prev_frame = None
    total_person_count = 0  # Initialize total person count

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Call the detection function
        frame, person_count, boxes, confidences, class_ids = detect_people(frame, net, output_layers, classes)

        # Update the total person count
        total_person_count += person_count

        # Track motion if the previous frame exists
        if prev_frame is not None:
            track_motion(prev_frame, frame, boxes)

        # Update the previous frame
        prev_frame = frame.copy()

        cv2.imshow("Motion and People Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
    return total_person_count  # Return total person count

if __name__ == "__main__":
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    detect_motion_and_people(classes)
