
import cv2
import time
import sys

def test_strategy(name, source, backend):
    print(f"\n--------------------------------------------------")
    print(f"Testing Strategy: {name}")
    print(f"Source: {source}")
    print(f"Backend: {backend}")
    print(f"--------------------------------------------------")

    try:
        cap = cv2.VideoCapture(source, backend)
        
        if not cap.isOpened():
            print("❌ Failed to open camera (cap.isOpened() is False).")
            return False

        print("📷 Camera opened. Attempting to read 10 frames...")
        
        success_count = 0
        req_frames = 10
        
        for i in range(req_frames):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                success_count += 1
                # print(f"  Frame {i+1}: OK ({frame.shape})")
                sys.stdout.write(".")
                sys.stdout.flush()
            else:
                print(f"\n  ❌ Frame {i+1}: Failed to read.")
                time.sleep(0.1)
        
        print("") # Newline

        if success_count == req_frames:
            print(f"\n✅ SUCCESS! Strategy '{name}' works perfectly.")
            print("\n>>> COPY THIS PYTHON CODE FOR YOUR SCRIPT: <<<")
            
            if backend == cv2.CAP_GSTREAMER:
                print(f'wrist_pipeline = "{source}"')
                print(f'wrist_cam = cv2.VideoCapture(wrist_pipeline, cv2.CAP_GSTREAMER)')
            elif backend == cv2.CAP_FFMPEG:
                print(f'wrist_source = "{source}"')
                print(f'wrist_cam = cv2.VideoCapture(wrist_source, cv2.CAP_FFMPEG)')
            else:
                print(f'wrist_source = "{source}"')
                print(f'wrist_cam = cv2.VideoCapture(wrist_source)')
            
            print("--------------------------------------------------\n")
            
            # Show video for verification
            print("🎥 Displaying video stream... Press 'q' to quit.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                cv2.imshow(f"Success: {name}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cv2.destroyAllWindows()
            cap.release()
            return True
        else:
            print(f"\n❌ unstable. Only read {success_count}/{req_frames} frames.")
            cap.release()
            return False

    except Exception as e:
        print(f"\n❌ Validation Exception: {e}")
        return False

def main():
    print("🕵️  Starting Camera Diagnosis Tool...")
    print("    Goal: Connect to StereoPi/Wrist Camera\n")

    strategies = [
        {
            "name": "Strategy A (Robust GStreamer)",
            "source": "udpsrc port=3000 buffer-size=5000000 ! h264parse ! avdec_h264 ! videoconvert ! appsink sync=false",
            "backend": cv2.CAP_GSTREAMER
        },
        {
            "name": "Strategy B (Simple GStreamer)",
            "source": "udpsrc port=3000 ! decodebin ! videoconvert ! appsink sync=false",
            "backend": cv2.CAP_GSTREAMER
        },
        {
            "name": "Strategy C (FFmpeg Backend)",
            "source": "udp://127.0.0.1:3000?fifo_size=5000000&overrun_nonfatal=1", 
            # Note: User said 10.42.0.1, but often localhost is safer if tunneling. 
            # I will assume user knows IP. User prompt: "udp://10.42.0.1:3000..."
            # Wait, if StereoPi is SENDING to the PC, the PC listens on 0.0.0.0 (any) or its own IP. 
            # `udpsrc` binds to local port. `udp://` in ffmpeg usually connects or listens?
            # FFmpeg `udp://hostname:port` -> connects. `udp://@:3000` -> listens.
            # User prompt says: "udp://10.42.0.1:3000". If this is the PC's IP, fine. 
            # If it's the StereoPi's IP, FFmpeg might try to connect TO it.
            # Usually StereoPi streams UDP to the PC. So PC should listen.
            # I'll stick to user's EXACT string to be safe, but "udp://@:3000" is usually correct for listening.
            # User provided: "udp://10.42.0.1:3000..." -> I will use that.
            "source": "udp://10.42.0.1:3000?fifo_size=5000000&overrun_nonfatal=1",
            "backend": cv2.CAP_FFMPEG
        },
        {
            "name": "Strategy D (HTTP Fallback)",
            "source": "http://10.42.0.100:5000/video_feed",
            "backend": cv2.CAP_ANY
        }
    ]

    for s in strategies:
        if test_strategy(s["name"], s["source"], s["backend"]):
            print("\n🎉 DIAGNOSIS COMPLETE: FOUND WORKING STRATEGY.")
            return

    print("\n❌ DIAGNOSIS FAILED. No suitable strategy found.")
    print("👉 Check firewall, physical connection, or if another process is holding the port.")

if __name__ == "__main__":
    main()
