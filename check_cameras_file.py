import cv2
import os

def take_snapshot(index):
    print(f"📸 Connecting to Camera {index}...")
    cap = cv2.VideoCapture(index)
    
    # Warm up camera
    if not cap.isOpened():
        print(f"❌ Failed to open Camera {index}")
        return

    # Read a few frames to let auto-exposure settle
    for _ in range(10):
        ret, frame = cap.read()

    if ret:
        filename = f"camera_{index}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ Saved image to: {os.path.abspath(filename)}")
    else:
        print(f"❌ Could not capture from Camera {index}")
    
    cap.release()

# Snap photos from both potential ports
take_snapshot(0)
take_snapshot(2)
print("\n👀 Go open these two image files in your file manager to identify them!")
