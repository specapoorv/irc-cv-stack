from onvif import ONVIFCamera
import cv2
import threading, queue, time

# --- CONFIGURATION ---
IP = "10.42.0.69"  # <--- YOUR IP
PORT = 80
USER = "admin"
PASS = "L26AF42F" # <--- YOUR PASSWORD

# RTSP Stream (Use subtype=1 for faster speed)
RTSP_URL = f"rtsp://{USER}:{PASS}@{IP}:554/cam/realmonitor?channel=1&subtype=1"


# ==========================================
# 1. NEW CLASS: BUFFERLESS CAPTURE (THE FIX)
# ==========================================
class BufferlessVideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0) # Force buffer to 0
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # Discard previous (old) frame
                except queue.Empty:
                    pass
            self.q.put(frame)             # Put new frame in queue

    def read(self):
        return self.q.get()

    def release(self):
        self.cap.release()

# ==========================================
# END OF NEW CLASS
# ==========================================

active_move = False

def setup_ptz():
    try:
        print(f"Connecting to ONVIF at {IP}...")
        mycam = ONVIFCamera(IP, PORT, USER, PASS)
        ptz = mycam.create_ptz_service()
        media = mycam.create_media_service()
        
        media_profile = media.GetProfiles()[0]
        token = media_profile.token
        
        print(f"✅ CAM Connected, Profile Token: {token}")
        return ptz, token
    except Exception as e:
        print(f"Connection Failed (mostly IP error): {e}")
        return None, None

def move(ptz, token, direction):
    global active_move
    if active_move: return # Don't spam commands
    
    request = ptz.create_type('ContinuousMove')
    request.ProfileToken = token
    
    if direction == 'Up':    request.Velocity = {'PanTilt': {'x': 0, 'y': 0.4}}
    elif direction == 'Down':  request.Velocity = {'PanTilt': {'x': 0, 'y': -0.4}}
    elif direction == 'Left':  request.Velocity = {'PanTilt': {'x': -0.4, 'y': 0}}
    elif direction == 'Right': request.Velocity = {'PanTilt': {'x': 0.4, 'y': 0}}
        
    try:
        ptz.ContinuousMove(request)
        active_move = True
        print(f"Moving {direction}")
    except Exception as e:
        print(f"Move Error: {e}")

def stop(ptz, token):
    global active_move
    try:
        ptz.Stop({'ProfileToken': token})
        active_move = False
        print("Stopped")
    except:
        pass

def main():
    print("--- ROVER ONVIF CONTROLLER (LOW LATENCY) ---")
    
    # 1. Setup Control
    ptz_service, token = setup_ptz()
    if not ptz_service:
        print("Could not connect controls. Exiting.")
        return

    # ==========================================
    # 2. CHANGED: USE NEW CLASS HERE
    # ==========================================
    print("Starting Video Stream...")
    
    # OLD LINE: cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    # NEW LINE:
    cap = BufferlessVideoCapture(RTSP_URL)
    
    # Give it a second to fill the buffer
    time.sleep(1.0) 

    print("Controls: W/A/S/D to Move | SPACE to Stop | Q to Quit")

    while True:
        # ==========================================
        # 3. CHANGED: READ LOGIC
        # ==========================================
        # OLD: ret, frame = cap.read()
        # NEW: (The class handles the 'ret' check internally)
        frame = cap.read()
        
        if frame is None:
            continue
        frame = cv2.flip(frame, -1)
        cv2.imshow("Rover Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'): move(ptz_service, token, 'Up')
        elif key == ord('w'): move(ptz_service, token, 'Down')
        elif key == ord('d'): move(ptz_service, token, 'Left')
        elif key == ord('a'): move(ptz_service, token, 'Right')
        elif key == 32:       stop(ptz_service, token) # SPACEBAR TO STOP
        elif key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
	
