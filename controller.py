from main import detect_motion_and_people

def main():
    print("Starting motion and people detection...")
    
    # Load the classes from coco.names
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Pass classes to detect_motion_and_people
    person_count = detect_motion_and_people(classes)
    print(f"Total number of people detected: {person_count}")

if __name__ == "__main__":
    main()
