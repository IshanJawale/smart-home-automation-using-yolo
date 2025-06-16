import cv2
import numpy as np
#import RPi.GPIO as GPIO

# RELAY_PIN = 33

# GPIO.setmode(GPIO.BOARD)
# GPIO.setwarnings(False)
# GPIO.setup(RELAY_PIN, GPIO.OUT)

def bulb(count):
    if count > 0:
        print("LIGHTS ON!!!")
        # GPIO.output(RELAY_PIN, GPIO.HIGH)
    else:
        print("LIGHTS OFF...")
        # GPIO.output(RELAY_PIN, GPIO.LOW)

# Function to detect people
def detect_people(frame, net, output_layers, classes):
    height, width = frame.shape[:2]

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    # Loop through the outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maxima Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    person_count = 0
    if len(indices) > 0:
        for i in indices:
            if isinstance(i, np.ndarray):
                i = i[0]
            box = boxes[i]
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            if label == "person":
                person_count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Print the number of people detected in the frame
    print(f"Number of people detected: {person_count}")
    cv2.putText(frame, f'Total Persons: {person_count}', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    bulb(person_count)
    return frame, person_count, boxes, confidences, class_ids
