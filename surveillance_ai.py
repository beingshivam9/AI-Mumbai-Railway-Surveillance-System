import cv2
import time
import threading
from ultralytics import YOLO
from playsound import playsound

# Load YOLO model
model = YOLO("yolov8m.pt")

# Open webcam
cap = cv2.VideoCapture(0)

bag_owners = {}  # Tracks which person owns which bag
bag_timers = {}  # Tracks when a bag is left unattended
alert_active = False  # Track if the alarm is playing

# Define class IDs
PERSON_CLASS = 0
BAG_CLASSES = [24, 26, 28]  # Backpack, Handbag, Suitcase

# Define unattended threshold
UNATTENDED_THRESHOLD = 5  # Time before an alert starts

def is_fully_outside(bag, person):
    """Check if the bag is completely outside the person’s bounding box."""
    bx1, by1, bx2, by2 = bag
    px1, py1, px2, py2 = person

    return bx2 < px1 or bx1 > px2 or by2 < py1 or by1 > py2

def play_alert():
    """Plays an alert sound."""
    global alert_active
    if not alert_active:
        alert_active = True
        playsound("alert_sound.mp3")
        alert_active = False

def stop_alert():
    """Stops the alarm immediately."""
    global alert_active
    alert_active = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  

    results = model(frame)
    current_time = time.time()
    detected_objects = []

    # Step 1: Detect objects
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.tolist()[0])  

            if class_id in [PERSON_CLASS] + BAG_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  
                conf = float(box.conf.tolist()[0])  
                detected_objects.append((x1, y1, x2, y2, class_id))

                color = (0, 255, 0)  
                label = model.names[class_id]  

                # Draw bounding boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Step 2: Identify people and bags
    people_positions = []
    bag_positions = []

    for obj in detected_objects:
        x1, y1, x2, y2, class_id = obj

        if class_id == PERSON_CLASS:  
            people_positions.append((x1, y1, x2, y2))
        
        elif class_id in BAG_CLASSES:  
            bag_positions.append((x1, y1, x2, y2))

    # Step 3: Assign bags to people
    new_bag_owners = {}
    for bag in bag_positions:
        for person in people_positions:
            if not is_fully_outside(bag, person):  
                new_bag_owners[bag] = person  # Assign bag to person

    # Step 4: Detect unattended bags
    for bag in bag_positions:
        if bag not in new_bag_owners:  # If no owner found
            if bag not in bag_timers:
                bag_timers[bag] = current_time  # Start timer
        else:
            if bag in bag_timers:
                del bag_timers[bag]  # Remove from alert if owner is back
                stop_alert()  # Immediately stop the alarm

    # Step 5: Check alert conditions
    for bag, start_time in list(bag_timers.items()):
        if current_time - start_time > UNATTENDED_THRESHOLD:
            print(f"🚨 ALERT: Unattended Bag Detected for {int(current_time - start_time)} sec")
            cv2.putText(frame, "ALERT: Unattended Bag!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if not alert_active:
                threading.Thread(target=play_alert, daemon=True).start()

    # Show result
    cv2.imshow("Surveillance System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
