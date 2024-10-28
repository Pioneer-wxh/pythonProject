import cv2
import numpy as np

def detect_laser_pointer():
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range of red color in HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])

        # Threshold the HSV image to get only red colors
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If contours are found, find the largest one
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Filter out small contours
            if radius > 5:
                # Draw the circle and centroid on the frame
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                cv2.circle(frame, center, 2, (0, 0, 255), -1)

                # Estimate distance (simple example, not accurate for real-world use)
                distance = estimate_distance(radius)
                cv2.putText(frame, f"Distance: {distance:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Laser Pointer Detection', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def estimate_distance(radius):
    # Simple linear model for distance estimation (not accurate for real-world use)
    # You can calibrate this function based on known distances and radii
    focal_length = 500  # Example focal length in pixels
    known_radius = 1  # Known radius of the laser dot in cm
    distance = (known_radius * focal_length) / radius
    return distance

if __name__ == "__main__":
    detect_laser_pointer()