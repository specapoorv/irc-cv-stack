from onvif import ONVIFCamera
import cv2
import threading
import queue
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from rclpy.qos import QoSProfile, ReliabilityPolicy

IP = "10.42.0.69"
PORT = 80
USER = "admin"
PASS = "L26AF42F"
RTSP_URL = f"rtsp://{USER}:{PASS}@{IP}:554/cam/realmonitor?channel=1&subtype=1"

class CamNode(Node):
    def __init__(self):
        super().__init__("cam")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, qos)
        self.updown = 0.0
        self.leftright = 0.0

    def joy_callback(self, joy: Joy):
        try:
            axes = joy.axes
            self.leftright = float(axes[6]) if len(axes) > 6 else 0.0
            self.updown = float(axes[7]) if len(axes) > 7 else 0.0
        except Exception as e:
            self.get_logger().error(f"Joy callback error: {e}")
            self.leftright = 0.0
            self.updown = 0.0

        thr = 0.5
        lr_state = 0
        ud_state = 0
        if self.leftright > thr:
            lr_state = 1
        elif self.leftright < -thr:
            lr_state = -1
        if self.updown > thr:
            ud_state = 1
        elif self.updown < -thr:
            ud_state = -1

        msg = ""
        if ud_state == 1:
            msg = "Moving up"
        elif ud_state == -1:
            msg = "Moving down"
        elif lr_state == 1:
            msg = "Moving left"
        elif lr_state == -1:
            msg = "Moving right"

        if msg:
            # (2) use .info()
            self.get_logger().info(msg)

class BufferlessVideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue(maxsize=1)
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                # keep newest frame only
                try:
                    if not self.q.empty():
                        try:
                            self.q.get_nowait()
                        except queue.Empty:
                            pass
                    self.q.put_nowait(frame)
                except queue.Full:
                    pass
            except Exception:
                time.sleep(0.01)
                continue

    def read(self, timeout=0.5):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        try:
            self.cap.release()
        except Exception:
            pass

active_move_lock = threading.Lock()
active_move = False

def setup_ptz():
    try:
        print(f"Connecting to ONVIF at {IP}...")
        mycam = ONVIFCamera(IP, PORT, USER, PASS)
        ptz = mycam.create_ptz_service()
        media = mycam.create_media_service()
        profiles = media.GetProfiles()
        if not profiles:
            raise RuntimeError("No media profiles returned")
        media_profile = profiles[0]
        token = media_profile.token
        print(f"CAM Connected, Profile Token: {token}")
        return ptz, token
    except Exception as e:
        print(f"Connection Failed: {e}")
        return None, None

def move(ptz, token, direction):
    global active_move
    if not direction:
        stop(ptz, token)
        return
    with active_move_lock:
        if active_move:
            return
        try:
            request = ptz.create_type('ContinuousMove')
            request.ProfileToken = token
            try:
                request.Velocity = {'PanTilt': {'x': 0.0, 'y': 0.0}}
            except Exception:
                request.Velocity = None
            if direction == 'Up':
                vel = {'PanTilt': {'x': 0.0, 'y': 0.4}}
            elif direction == 'Down':
                vel = {'PanTilt': {'x': 0.0, 'y': -0.4}}
            elif direction == 'Left':
                vel = {'PanTilt': {'x': -0.4, 'y': 0.0}}
            elif direction == 'Right':
                vel = {'PanTilt': {'x': 0.4, 'y': 0.0}}
            else:
                stop(ptz, token)
                return
            
            try:
                request.Velocity = vel
                ptz.ContinuousMove(request)
            except Exception:
                try:
                    request.__setattr__('Velocity', vel)
                    ptz.ContinuousMove(request)
                except Exception as e:
                    print(f"Move Error (fallback): {e}")
                    return

            active_move = True
            print(f"Moving {direction}")
        except Exception as e:
            print(f"Move Error: {e}")

def stop(ptz, token):
    global active_move
    try:
        ptz.Stop({'ProfileToken': token})
    except Exception:
        pass
    with active_move_lock:
        active_move = False
    print("Stopped")

def main(args=None):
    rclpy.init(args=args)
    node = CamNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    ptz_service, token = setup_ptz()
    if not ptz_service:
        print("Could not connect controls. Exiting.")
        rclpy.shutdown()
        return

    print("Starting Video Stream...")
    cap = BufferlessVideoCapture(RTSP_URL)
    time.sleep(1.0)

    try:
        while True:
            frame = cap.read(timeout=0.5)
            if frame is None:
                continue
            frame = cv2.flip(frame, -1)
            cv2.imshow("Rover Feed", frame)
            key = cv2.waitKey(1) & 0xFF

            if node.updown == -1:
                move(ptz_service, token, 'Up')
            elif node.updown == 1:
                move(ptz_service, token, 'Down')
            elif node.leftright == -1:
                move(ptz_service, token, 'Left')
            elif node.leftright == 1:
                move(ptz_service, token, 'Right')
            else:
                stop(ptz_service, token)

            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == "__main__":
    main()
