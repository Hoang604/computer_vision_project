import cv2
import mediapipe as mp
import numpy as np
from dtaidistance import dtw
import time
import os

# --- Khởi tạo các đối tượng MediaPipe ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Các hằng số và cấu hình ---
EXPERT_VIDEO_PATH = "data/converted_mp4/v1.mp4"
USER_VIDEO_PATH = "data/converted_mp4/v1.mp4"
OUTPUT_VIDEO_PATH = "output/v1.mp4" 

PANEL_WIDTH = 640
PANEL_HEIGHT = 480

COLOR_EXPERT_SKELETON = (255, 0, 0)
COLOR_USER_SKELETON = (0, 0, 255)
COLOR_EXPERT_ANGLES = (255, 128, 0)
COLOR_USER_ANGLES = (0, 128, 255)
COLOR_OVERLAY_EXPERT = (255, 100, 100)
COLOR_OVERLAY_USER = (100, 100, 255)

LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
LEFT_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value
LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
RIGHT_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW.value
RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value
LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value

# --- Hàm tiện ích (calculate_angle, extract_video_data, get_dtw_feature_series, draw_skeleton_and_angles GIỮ NGUYÊN) ---
def calculate_angle(p1_coords, p2_coords, p3_coords):
    p1 = np.array(p1_coords[:2])
    p2 = np.array(p2_coords[:2])
    p3 = np.array(p3_coords[:2])
    vector1 = p1 - p2
    vector2 = p3 - p2
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    if norm_product == 0: return 0.0
    cosine_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)

def extract_video_data(video_path, pose_detector):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return [], [], [], 0
    frames = []
    screen_landmarks_series = []
    world_landmarks_series = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Detected FPS for {video_path}: {fps}")
    while cap.isOpened():
        success, image = cap.read()
        if not success: break
        frames.append(image.copy())
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose_detector.process(image_rgb)
        image_rgb.flags.writeable = True
        if results.pose_landmarks:
            screen_landmarks_series.append([[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark])
            if results.pose_world_landmarks:
                world_landmarks_series.append([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_world_landmarks.landmark])
            else: world_landmarks_series.append(None)
        else:
            screen_landmarks_series.append(None)
            world_landmarks_series.append(None)
    cap.release()
    print(f"Extracted {len(frames)} frames from {video_path} at {fps:.2f} FPS.")
    return frames, screen_landmarks_series, world_landmarks_series, fps

def get_dtw_feature_series(landmarks_series, landmark_index, coord_index=1):
    series = []
    for frame_landmarks in landmarks_series:
        if frame_landmarks and len(frame_landmarks) > landmark_index and frame_landmarks[landmark_index] is not None:
             # Check visibility if available (index 2 for screen, 3 for world)
            visibility_idx = 3 if len(frame_landmarks[landmark_index]) == 4 else 2 # World landmarks have 4 components
            if frame_landmarks[landmark_index][visibility_idx] > 0.3:
                series.append(frame_landmarks[landmark_index][coord_index])
            elif series: series.append(series[-1])
        elif series: series.append(series[-1])
    return np.array(series, dtype=np.double)

def draw_skeleton_and_angles(image, landmarks_2d, color_skeleton, color_angles, panel_width, panel_height):
    if landmarks_2d:
        pixel_landmarks = []
        valid_landmarks_for_angle = {}
        for i, (x_norm, y_norm, visibility) in enumerate(landmarks_2d):
            if visibility < 0.3:
                pixel_landmarks.append(None)
                continue
            px = int(x_norm * panel_width)
            py = int(y_norm * panel_height)
            pixel_landmarks.append((px, py))
            valid_landmarks_for_angle[i] = (px,py)
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if pixel_landmarks[start_idx] and pixel_landmarks[end_idx]:
                 cv2.line(image, pixel_landmarks[start_idx], pixel_landmarks[end_idx], color_skeleton, 2)
        for lm_px in pixel_landmarks:
            if lm_px: cv2.circle(image, lm_px, 3, color_skeleton, -1)
        
        # Left Elbow
        if all(idx in valid_landmarks_for_angle for idx in [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]):
            angle = calculate_angle(landmarks_2d[LEFT_SHOULDER], landmarks_2d[LEFT_ELBOW], landmarks_2d[LEFT_WRIST])
            cv2.putText(image, f"{angle:.1f}", (valid_landmarks_for_angle[LEFT_ELBOW][0] + 10, valid_landmarks_for_angle[LEFT_ELBOW][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_angles, 2)
        # Right Elbow
        if all(idx in valid_landmarks_for_angle for idx in [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]):
            angle = calculate_angle(landmarks_2d[RIGHT_SHOULDER], landmarks_2d[RIGHT_ELBOW], landmarks_2d[RIGHT_WRIST])
            cv2.putText(image, f"{angle:.1f}", (valid_landmarks_for_angle[RIGHT_ELBOW][0] - 50, valid_landmarks_for_angle[RIGHT_ELBOW][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_angles, 2)


# --- HÀM draw_world_skeleton_on_panel ĐƯỢC CẬP NHẬT ---
def draw_world_skeleton_on_panel(panel, world_landmarks, color, panel_width, panel_height):
    """Vẽ world landmarks lên một panel 2D, ĐÃ SỬA LỖI LỘN NGƯỢC."""
    if not world_landmarks:
        return

    # Chỉ lấy các landmark hợp lệ (có visibility tốt)
    # world_landmarks có dạng [x, y, z, visibility]
    # MediaPipe Y: dương là hướng lên. OpenCV Y: dương là hướng xuống.
    valid_lms_data = []
    for lm in world_landmarks:
        if lm[3] > 0.3: # Check visibility
            # X giữ nguyên, Y đảo ngược cho hệ tọa độ màn hình, Z không dùng để vẽ 2D
            valid_lms_data.append([lm[0], -lm[1], lm[2]]) # Đảo ngược Y ở đây

    if not valid_lms_data: return

    xs = [lm[0] for lm in valid_lms_data]
    ys = [lm[1] for lm in valid_lms_data] # Y ở đây đã được đảo ngược (-world_y)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys) # min_y giờ là điểm "cao nhất" trên màn hình (Y nhỏ nhất sau khi đảo)
                                    # max_y là điểm "thấp nhất" trên màn hình (Y lớn nhất sau khi đảo)

    range_x = max_x - min_x if max_x > min_x else 1.0
    range_y = max_y - min_y if max_y > min_y else 1.0
    
    scale_x = (panel_width * 0.8) / range_x  # Để lại 20% lề
    scale_y = (panel_height * 0.8) / range_y
    scale = min(scale_x, scale_y) # Giữ tỷ lệ khung hình của skeleton

    # Tính toán tâm của skeleton trong hệ tọa độ đã biến đổi (Y đã đảo)
    center_transformed_x = (min_x + max_x) / 2
    center_transformed_y = (min_y + max_y) / 2
    
    # ... (min_x, max_x, min_y, max_y, range_x, range_y, scale_x, scale_y, scale đã được tính)
    # ... (center_transformed_x, center_transformed_y đã được tính từ valid_lms_data với Y đã đảo)

    pixel_landmarks = {}
    for i, lm_3d_full in enumerate(world_landmarks):
        if lm_3d_full[3] < 0.3: # Visibility
            continue

        world_x, world_y_original, _, _ = lm_3d_full

        # Giữ nguyên transformed_y vì center_transformed_y được tính dựa trên nó
        transformed_y = -world_y_original 

        # Tính toán px, căn giữa vào panel_width / 2
        px = int((world_x - center_transformed_x) * scale + (panel_width / 2))
        
        # Tính toán py:
        # Logic hiện tại của bạn (đã được phân tích ở trên) là:
        # py = int((transformed_y - center_transformed_y) * scale + (panel_height / 2))
        # Nếu điều này gây lộn ngược, hãy thử đảo ngược thành phần scale:
        py = int( (panel_height / 2) - ((transformed_y - center_transformed_y) * scale) )
        # Hoặc, nếu bạn muốn giữ nguyên cấu trúc offset_y:
        # offset_y_corrected = (panel_height / 2) + (center_transformed_y * scale) # Dấu + thay vì -
        # py = int(transformed_y * scale + offset_y_corrected) # Điều này sẽ không đúng nếu transformed_y đã âm
        # Vì vậy, cách tính trực tiếp với (panel_height / 2) - (...) thường rõ ràng hơn khi cần đảo ngược.

        pixel_landmarks[i] = (px, py)

    # Vẽ các đường nối
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx in pixel_landmarks and end_idx in pixel_landmarks:
            cv2.line(panel, pixel_landmarks[start_idx], pixel_landmarks[end_idx], color, 2)
    # Vẽ các điểm
    for lm_px in pixel_landmarks.values(): # Duyệt qua values của dict
        cv2.circle(panel, lm_px, 3, color, -1)


# --- Hàm chính (ĐÃ CẬP NHẬT ĐỂ KIỂM SOÁT TỐC ĐỘ XEM TRỰC TIẾP) ---
def main():
    pose = mp_pose.Pose(
        static_image_mode=False, model_complexity=1, smooth_landmarks=True,
        enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    print("Extracting expert video data...")
    expert_frames, expert_s_lms_series, expert_w_lms_series, expert_fps = extract_video_data(EXPERT_VIDEO_PATH, pose)
    print("Extracting user video data...")
    user_frames, user_s_lms_series, user_w_lms_series, user_fps = extract_video_data(USER_VIDEO_PATH, pose)

    if not expert_frames or not user_frames:
        print("Error: Could not process one or both videos.")
        return
    
    output_fps = 60
    desired_frame_duration = 1.0 / output_fps # Thời gian hiển thị mỗi frame (giây)

    valid_expert_indices = [i for i, x in enumerate(expert_s_lms_series) if x and expert_w_lms_series[i]]
    valid_user_indices = [i for i, x in enumerate(user_s_lms_series) if x and user_w_lms_series[i]]

    if not valid_expert_indices or not valid_user_indices:
        print("Not enough valid landmarks for DTW.")
        return

    dtw_expert_series = get_dtw_feature_series([expert_w_lms_series[i] for i in valid_expert_indices], mp_pose.PoseLandmark.NOSE.value, 1)
    dtw_user_series = get_dtw_feature_series([user_w_lms_series[i] for i in valid_user_indices], mp_pose.PoseLandmark.NOSE.value, 1)

    if len(dtw_expert_series) < 2 or len(dtw_user_series) < 2:
        print(f"Not enough data for DTW. Expert: {len(dtw_expert_series)}, User: {len(dtw_user_series)}")
        return

    print("Performing Dynamic Time Warping...")
    warping_path = dtw.warping_path(dtw_expert_series, dtw_user_series, window=int(max(len(dtw_expert_series), len(dtw_user_series)) * 0.35), use_c=True, psi=1) # Thêm psi để nới lỏng hơn một chút
    print(f"DTW complete. Warping path length: {len(warping_path)}")

    output_frame_width = PANEL_WIDTH * 3
    output_frame_height = PANEL_HEIGHT
    output_dir = os.path.dirname(OUTPUT_VIDEO_PATH)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, output_fps, (output_frame_width, output_frame_height))
    if not video_writer.isOpened():
        print(f"Lỗi: Không thể mở VideoWriter cho file {OUTPUT_VIDEO_PATH}")
        video_writer = None
    else: print(f"Đang lưu video vào: {OUTPUT_VIDEO_PATH} với {output_fps:.2f} FPS")

    cv2.namedWindow("Gym Form Analysis", cv2.WINDOW_NORMAL)
    display_h, display_w = PANEL_HEIGHT, PANEL_WIDTH

    for idx_expert_in_dtw_path, idx_user_in_dtw_path in warping_path:
        frame_start_time = time.time() # Bắt đầu tính thời gian xử lý frame

        actual_expert_frame_idx = valid_expert_indices[idx_expert_in_dtw_path]
        actual_user_frame_idx = valid_user_indices[idx_user_in_dtw_path]

        if actual_expert_frame_idx >= len(expert_frames) or actual_user_frame_idx >= len(user_frames):
            continue

        expert_frame_orig = expert_frames[actual_expert_frame_idx]
        user_frame_orig = user_frames[actual_user_frame_idx]
        expert_s_lms = expert_s_lms_series[actual_expert_frame_idx]
        user_s_lms = user_s_lms_series[actual_user_frame_idx]
        expert_w_lms = expert_w_lms_series[actual_expert_frame_idx]
        user_w_lms = user_w_lms_series[actual_user_frame_idx]

        panel_expert = cv2.resize(expert_frame_orig.copy(), (display_w, display_h))
        panel_user = cv2.resize(user_frame_orig.copy(), (display_w, display_h))
        panel_overlay = np.zeros((display_h, display_w, 3), dtype=np.uint8)

        if expert_s_lms:
            draw_skeleton_and_angles(panel_expert, expert_s_lms, COLOR_EXPERT_SKELETON, COLOR_EXPERT_ANGLES, display_w, display_h)
        if user_s_lms:
            draw_skeleton_and_angles(panel_user, user_s_lms, COLOR_USER_SKELETON, COLOR_USER_ANGLES, display_w, display_h)
        
        if expert_w_lms:
             draw_world_skeleton_on_panel(panel_overlay, expert_w_lms, COLOR_OVERLAY_EXPERT, display_w, display_h)
        if user_w_lms:
             draw_world_skeleton_on_panel(panel_overlay, user_w_lms, COLOR_OVERLAY_USER, display_w, display_h)

        combined_display = np.hstack((panel_expert, panel_user, panel_overlay))
        cv2.imshow("Gym Form Analysis", combined_display)

        if video_writer and video_writer.isOpened():
            video_writer.write(combined_display)

        # Kiểm soát tốc độ xem trực tiếp
        frame_processing_time = time.time() - frame_start_time
        wait_time_seconds = desired_frame_duration - frame_processing_time
        wait_time_ms = int(max(1, wait_time_seconds * 1000)) # Chờ ít nhất 1ms

        if cv2.waitKey(wait_time_ms) & 0xFF == ord('q'):
            break
    
    print("Closing program.")
    pose.close()
    if video_writer and video_writer.isOpened():
        video_writer.release()
        print(f"Video saved to {OUTPUT_VIDEO_PATH}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.exists(EXPERT_VIDEO_PATH):
        print(f"Error: Expert video file not found at: {EXPERT_VIDEO_PATH}")
    elif not os.path.exists(USER_VIDEO_PATH):
        print(f"Error: User video file not found at: {USER_VIDEO_PATH}")
    else:
        main()
