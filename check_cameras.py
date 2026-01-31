import cv2

def test_cam(index, name):
    print(f"Opening Camera {index} ({name})...")
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"❌ Failed to open Camera {index}")
        return
    
    ret, frame = cap.read()
    if ret:
        cv2.imshow(f"Camera {index} - IS THIS THE {name}?", frame)
        cv2.waitKey(2000) # Show for 2 seconds
        cap.release()
    else:
        print(f"❌ Could not read frame from Camera {index}")

# Check 0 and 2
test_cam(0, "FIRST CAM")
test_cam(2, "SECOND CAM")

cv2.destroyAllWindows()
print("\nDONE! Which one was the Wrist Camera?")
