import cv2
import mediapipe as mp
import numpy as np
from dtaidistance import dtw
import time
import os

# --- Initialize MediaPipe objects ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Constants and Configurations ---
EXPERT_VIDEO_PATH = "data/v1.webm"  # Path to the expert's video
USER_VIDEO_PATH = "data/v1.webm"    # Path to the user's video
OUTPUT_VIDEO_PATH = "output/v1.mp4" # Path to save the output video

PANEL_WIDTH = 640  # Width of each panel in the output display
PANEL_HEIGHT = 480 # Height of each panel in the output display

COLOR_EXPERT_SKELETON = (255, 0, 0)    # Blue for expert skeleton (BGR)
COLOR_USER_SKELETON = (0, 0, 255)      # Red for user skeleton (BGR)
COLOR_EXPERT_ANGLES = (255, 128, 0)  # Light blue for expert angles
COLOR_USER_ANGLES = (0, 128, 255)    # Orange for user angles
COLOR_OVERLAY_EXPERT = (255, 100, 100) # Lighter blue for expert overlay skeleton
COLOR_OVERLAY_USER = (100, 100, 255)   # Lighter red for user overlay skeleton

# Pose landmark constants for easier access
LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
LEFT_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value
LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
RIGHT_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW.value
RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value
LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value

# --- Utility Functions ---
def calculate_angle(p1_coords, p2_coords, p3_coords):
    """
    Calculates the angle between three 2D points (p1-p2-p3).

    Args:
        p1_coords (list or tuple): Coordinates of the first point [x, y, ...].
        p2_coords (list or tuple): Coordinates of the second point (vertex) [x, y, ...].
        p3_coords (list or tuple): Coordinates of the third point [x, y, ...].

    Returns:
        float: The angle in degrees. Returns 0.0 if calculation is not possible.
    """
    p1 = np.array(p1_coords[:2])  # Take only x, y
    p2 = np.array(p2_coords[:2])  # Take only x, y
    p3 = np.array(p3_coords[:2])  # Take only x, y

    vector1 = p1 - p2
    vector2 = p3 - p2

    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)

    if norm_product == 0:  # Avoid division by zero
        return 0.0

    cosine_angle = np.clip(dot_product / norm_product, -1.0, 1.0)  # Ensure value is within arccos domain
    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)

def extract_video_data(video_path, pose_detector):
    """
    Extracts frames, screen landmarks, and world landmarks from a video file.

    Args:
        video_path (str): Path to the video file.
        pose_detector (mediapipe.solutions.pose.Pose): Initialized MediaPipe Pose detector.

    Returns:
        tuple: A tuple containing:
            - frames (list): List of BGR frames from the video.
            - screen_landmarks_series (list): List of screen landmark series (list of [x, y, visibility] per frame).
            - world_landmarks_series (list): List of world landmark series (list of [x, y, z, visibility] per frame).
            - fps (float): Frames per second of the video.
    """
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
        if not success:
            break
        frames.append(image.copy()) # Store a copy of the frame

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False # To improve performance, optionally mark the image as not writeable

        # Process the image and detect pose
        results = pose_detector.process(image_rgb)

        image_rgb.flags.writeable = True # Make the image writeable again

        if results.pose_landmarks:
            screen_landmarks_series.append([[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark])
            if results.pose_world_landmarks:
                world_landmarks_series.append([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_world_landmarks.landmark])
            else:
                world_landmarks_series.append(None) # Append None if world landmarks are not available for this frame
        else:
            screen_landmarks_series.append(None) # Append None if no landmarks are detected
            world_landmarks_series.append(None)

    cap.release()
    print(f"Extracted {len(frames)} frames from {video_path} at {fps:.2f} FPS.")
    return frames, screen_landmarks_series, world_landmarks_series, fps

def get_dtw_feature_series(landmarks_series, landmark_index, coord_index=1):
    """
    Extracts a specific coordinate from a specific landmark over a series of frames for DTW.

    Args:
        landmarks_series (list): A list of landmark sets (each set is for a frame).
                                 Each landmark set can be screen or world landmarks.
        landmark_index (int): The index of the landmark to extract (e.g., mp_pose.PoseLandmark.NOSE.value).
        coord_index (int): The index of the coordinate to extract (0 for x, 1 for y, 2 for z). Default is 1 (y).

    Returns:
        numpy.ndarray: A 1D array of the extracted feature, suitable for DTW.
                       Handles missing landmarks by repeating the last known value.
    """
    series = []
    for frame_landmarks in landmarks_series:
        if frame_landmarks and len(frame_landmarks) > landmark_index and frame_landmarks[landmark_index] is not None:
            # Determine visibility index based on landmark type (screen vs world)
            # World landmarks have 4 components [x, y, z, visibility], screen has 3 [x, y, visibility]
            visibility_idx = 3 if len(frame_landmarks[landmark_index]) == 4 else 2

            if frame_landmarks[landmark_index][visibility_idx] > 0.3: # Check visibility
                series.append(frame_landmarks[landmark_index][coord_index])
            elif series: # If current landmark is not visible but we have previous values
                series.append(series[-1]) # Repeat last known value
        elif series: # If no landmarks for this frame but we have previous values
            series.append(series[-1]) # Repeat last known value
    return np.array(series, dtype=np.double)


def draw_skeleton_and_angles(image, landmarks_2d, color_skeleton, color_angles, panel_width, panel_height):
    """
    Draws the pose skeleton and specific joint angles on an image.

    Args:
        image (numpy.ndarray): The image (panel) to draw on.
        landmarks_2d (list): List of 2D screen landmarks [[x_norm, y_norm, visibility], ...].
        color_skeleton (tuple): BGR color for the skeleton lines and points.
        color_angles (tuple): BGR color for the angle text.
        panel_width (int): Width of the panel (for converting normalized coords to pixels).
        panel_height (int): Height of the panel.
    """
    if landmarks_2d:
        pixel_landmarks = []
        valid_landmarks_for_angle = {} # Store pixel coordinates of visible landmarks for angle calculation

        # Convert normalized landmarks to pixel coordinates
        for i, (x_norm, y_norm, visibility) in enumerate(landmarks_2d):
            if visibility < 0.3: # Skip landmarks with low visibility
                pixel_landmarks.append(None)
                continue
            px = int(x_norm * panel_width)
            py = int(y_norm * panel_height)
            pixel_landmarks.append((px, py))
            valid_landmarks_for_angle[i] = (px, py) # Store if visible

        # Draw connections
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if pixel_landmarks[start_idx] and pixel_landmarks[end_idx]:
                 cv2.line(image, pixel_landmarks[start_idx], pixel_landmarks[end_idx], color_skeleton, 2)

        # Draw landmark points
        for lm_px in pixel_landmarks:
            if lm_px:
                cv2.circle(image, lm_px, 3, color_skeleton, -1)

        # Calculate and draw angles
        # Left Elbow
        if all(idx in valid_landmarks_for_angle for idx in [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]):
            angle = calculate_angle(landmarks_2d[LEFT_SHOULDER], landmarks_2d[LEFT_ELBOW], landmarks_2d[LEFT_WRIST])
            cv2.putText(image, f"{angle:.1f}", (valid_landmarks_for_angle[LEFT_ELBOW][0] + 10, valid_landmarks_for_angle[LEFT_ELBOW][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_angles, 2)
        # Right Elbow
        if all(idx in valid_landmarks_for_angle for idx in [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]):
            angle = calculate_angle(landmarks_2d[RIGHT_SHOULDER], landmarks_2d[RIGHT_ELBOW], landmarks_2d[RIGHT_WRIST])
            cv2.putText(image, f"{angle:.1f}", (valid_landmarks_for_angle[RIGHT_ELBOW][0] - 50, valid_landmarks_for_angle[RIGHT_ELBOW][1]), # Adjusted position for right elbow
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_angles, 2)


def draw_world_skeleton_on_panel(panel, world_landmarks, color, panel_width, panel_height):
    """
    Draws 3D world landmarks onto a 2D panel, scaled and centered.
    Corrects the inverted Y-axis issue.

    Args:
        panel (numpy.ndarray): The 2D image panel (e.g., a black canvas) to draw on.
        world_landmarks (list): List of 3D world landmarks [[x, y, z, visibility], ...].
        color (tuple): BGR color for the skeleton.
        panel_width (int): Width of the drawing panel.
        panel_height (int): Height of the drawing panel.
    """
    if not world_landmarks:
        return

    # Filter valid landmarks (good visibility) and transform coordinates
    # MediaPipe's Y is positive upwards. OpenCV's Y is positive downwards.
    # We need to invert Y for correct screen display.
    valid_lms_data = []
    for lm in world_landmarks:
        if lm[3] > 0.3: # Check visibility (index 3 for world_landmarks)
            # X remains, Y is inverted for screen coordinates, Z is not used for 2D drawing
            valid_lms_data.append([lm[0], -lm[1], lm[2]]) # Invert Y here

    if not valid_lms_data:
        return

    # Extract X and Y coordinates (Y is already inverted)
    xs = [lm[0] for lm in valid_lms_data]
    ys = [lm[1] for lm in valid_lms_data] # Y here is already -world_y

    # Determine the bounding box of the skeleton in the transformed coordinate system
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys) # min_y is now the "highest" point on screen (smallest Y after inversion)
                                    # max_y is the "lowest" point on screen (largest Y after inversion)

    # Calculate the range of coordinates
    range_x = max_x - min_x if max_x > min_x else 1.0 # Avoid division by zero
    range_y = max_y - min_y if max_y > min_y else 1.0

    # Calculate scaling factor to fit skeleton within 80% of panel dimensions, preserving aspect ratio
    scale_x = (panel_width * 0.8) / range_x
    scale_y = (panel_height * 0.8) / range_y
    scale = min(scale_x, scale_y) # Use the smaller scale to maintain aspect ratio

    # Calculate the center of the skeleton in the transformed coordinate system
    center_transformed_x = (min_x + max_x) / 2
    center_transformed_y = (min_y + max_y) / 2 # Center of the already Y-inverted coordinates

    pixel_landmarks = {} # Dictionary to store pixel coordinates of landmarks
    for i, lm_3d_full in enumerate(world_landmarks): # Iterate through original world_landmarks to get indices
        if lm_3d_full[3] < 0.3: # Skip low visibility landmarks
            continue

        world_x, world_y_original, _, _ = lm_3d_full # Original world Y

        # Use the same Y transformation as for bounding box calculation
        transformed_y = -world_y_original

        # Scale and translate to panel coordinates
        # Center the skeleton in the panel
        px = int((world_x - center_transformed_x) * scale + (panel_width / 2))

        # For py, we want to map the transformed_y range to the panel height.
        # The center_transformed_y is the midpoint of the (inverted) y-values.
        # We map this center to panel_height / 2.
        # A point `transformed_y` is `(transformed_y - center_transformed_y)` away from this center.
        # Scale this difference and add to `panel_height / 2`.
        # Since MediaPipe's Y is up and screen Y is down, and we've already inverted Y
        # (so larger transformed_y means lower on the body, which should be higher on screen if not for this logic),
        # the correction `(panel_height / 2) - ((transformed_y - center_transformed_y) * scale)` or
        # `(panel_height / 2) + (center_transformed_y - transformed_y) * scale` ensures correct orientation.
        # The provided solution `py = int( (panel_height / 2) - ((transformed_y - center_transformed_y) * scale) )`
        # aims to correct the up-down flip.
        # Let's re-verify:
        # If transformed_y > center_transformed_y (point is "lower" in original MP world coords, "higher" in our inverted Y system),
        # then (transformed_y - center_transformed_y) is positive.
        # (panel_height / 2) - (positive_value * scale) will place it *above* the panel center. This seems correct.
        # If transformed_y < center_transformed_y (point is "higher" in original MP, "lower" in inverted Y),
        # then (transformed_y - center_transformed_y) is negative.
        # (panel_height / 2) - (negative_value * scale) = (panel_height / 2) + (positive_value * scale), placing it *below* panel center. Correct.
        py = int( (panel_height / 2) - ((transformed_y - center_transformed_y) * scale) )


        pixel_landmarks[i] = (px, py)

    # Draw connections
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx in pixel_landmarks and end_idx in pixel_landmarks:
            cv2.line(panel, pixel_landmarks[start_idx], pixel_landmarks[end_idx], color, 2)

    # Draw landmark points
    for lm_px in pixel_landmarks.values(): # Iterate through the values (coordinates) of the dictionary
        cv2.circle(panel, lm_px, 3, color, -1)


# --- Main Function (Updated for live view speed control) ---
def main():
    """
    Main function to perform pose analysis by comparing user video to an expert video.
    It extracts pose landmarks, uses Dynamic Time Warping (DTW) to align the movements,
    and displays the expert, user, and an overlay of their skeletons side-by-side.
    The output can also be saved to a video file.
    """
    # Initialize Pose model
    pose = mp_pose.Pose(
        static_image_mode=False,        # Process video stream
        model_complexity=1,             # Model complexity (0, 1, or 2)
        smooth_landmarks=True,          # Reduce jitter
        enable_segmentation=False,      # Disable segmentation mask
        min_detection_confidence=0.5,   # Minimum confidence for person detection
        min_tracking_confidence=0.5)    # Minimum confidence for landmark tracking

    print("Extracting expert video data...")
    expert_frames, expert_s_lms_series, expert_w_lms_series, expert_fps = extract_video_data(EXPERT_VIDEO_PATH, pose)
    print("Extracting user video data...")
    user_frames, user_s_lms_series, user_w_lms_series, user_fps = extract_video_data(USER_VIDEO_PATH, pose)

    if not expert_frames or not user_frames:
        print("Error: Could not process one or both videos.")
        return

    output_fps = 60  # Desired FPS for the output video and live preview
    desired_frame_duration = 1.0 / output_fps # Target time per frame in seconds

    # Filter out frames where landmarks (both screen and world) were not detected
    valid_expert_indices = [i for i, (s_lm, w_lm) in enumerate(zip(expert_s_lms_series, expert_w_lms_series)) if s_lm and w_lm]
    valid_user_indices = [i for i, (s_lm, w_lm) in enumerate(zip(user_s_lms_series, user_w_lms_series)) if s_lm and w_lm]


    if not valid_expert_indices or not valid_user_indices:
        print("Not enough valid landmarks for DTW after filtering.")
        return

    # Prepare feature series for DTW using Y-coordinate of the NOSE from world landmarks
    # Only use landmarks from valid frames
    dtw_expert_series = get_dtw_feature_series([expert_w_lms_series[i] for i in valid_expert_indices], mp_pose.PoseLandmark.NOSE.value, 1) # 1 for Y-coord
    dtw_user_series = get_dtw_feature_series([user_w_lms_series[i] for i in valid_user_indices], mp_pose.PoseLandmark.NOSE.value, 1)   # 1 for Y-coord

    if len(dtw_expert_series) < 2 or len(dtw_user_series) < 2: # DTW needs at least 2 points
        print(f"Not enough data points for DTW after feature extraction. Expert series: {len(dtw_expert_series)}, User series: {len(dtw_user_series)}")
        return

    print("Performing Dynamic Time Warping...")
    # DTW window size can be adjusted; 35% of max length is a common heuristic.
    # psi parameter can relax constraints for warping path, useful for sequences with larger speed variations.
    warping_path = dtw.warping_path(dtw_expert_series, dtw_user_series,
                                    window=int(max(len(dtw_expert_series), len(dtw_user_series)) * 0.35),
                                    use_c=True, # Use C implementation for speed
                                    psi=1)      # Relaxation parameter for start/end of warping path
    print(f"DTW complete. Warping path length: {len(warping_path)}")

    # Setup for output video
    output_frame_width = PANEL_WIDTH * 3  # Three panels: expert, user, overlay
    output_frame_height = PANEL_HEIGHT
    output_dir = os.path.dirname(OUTPUT_VIDEO_PATH)
    if output_dir and not os.path.exists(output_dir): # Create output directory if it doesn't exist
        os.makedirs(output_dir)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4 video
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, output_fps, (output_frame_width, output_frame_height))

    if not video_writer.isOpened():
        print(f"Error: Could not open VideoWriter for file {OUTPUT_VIDEO_PATH}")
        video_writer = None # Set to None to avoid errors later
    else:
        print(f"Saving video to: {OUTPUT_VIDEO_PATH} at {output_fps:.2f} FPS")

    cv2.namedWindow("Gym Form Analysis", cv2.WINDOW_NORMAL) # Create a resizable window
    display_h, display_w = PANEL_HEIGHT, PANEL_WIDTH # Dimensions for individual display panels

    # Iterate through the DTW warping path to align and display frames
    for idx_expert_in_dtw_path, idx_user_in_dtw_path in warping_path:
        frame_start_time = time.time() # For controlling playback speed

        # Map DTW path indices back to original frame indices (from the valid, filtered lists)
        actual_expert_frame_idx = valid_expert_indices[idx_expert_in_dtw_path]
        actual_user_frame_idx = valid_user_indices[idx_user_in_dtw_path]

        # Boundary checks for safety, though DTW path should be within bounds of valid_indices
        if actual_expert_frame_idx >= len(expert_frames) or actual_user_frame_idx >= len(user_frames):
            continue

        # Get original frames and corresponding landmarks
        expert_frame_orig = expert_frames[actual_expert_frame_idx]
        user_frame_orig = user_frames[actual_user_frame_idx]
        expert_s_lms = expert_s_lms_series[actual_expert_frame_idx]
        user_s_lms = user_s_lms_series[actual_user_frame_idx]
        expert_w_lms = expert_w_lms_series[actual_expert_frame_idx]
        user_w_lms = user_w_lms_series[actual_user_frame_idx]

        # Create panels for display
        panel_expert = cv2.resize(expert_frame_orig.copy(), (display_w, display_h))
        panel_user = cv2.resize(user_frame_orig.copy(), (display_w, display_h))
        panel_overlay = np.zeros((display_h, display_w, 3), dtype=np.uint8) # Black panel for overlay

        # Draw skeletons and angles on respective panels
        if expert_s_lms:
            draw_skeleton_and_angles(panel_expert, expert_s_lms, COLOR_EXPERT_SKELETON, COLOR_EXPERT_ANGLES, display_w, display_h)
        if user_s_lms:
            draw_skeleton_and_angles(panel_user, user_s_lms, COLOR_USER_SKELETON, COLOR_USER_ANGLES, display_w, display_h)

        # Draw world skeletons on the overlay panel
        if expert_w_lms:
             draw_world_skeleton_on_panel(panel_overlay, expert_w_lms, COLOR_OVERLAY_EXPERT, display_w, display_h)
        if user_w_lms:
             draw_world_skeleton_on_panel(panel_overlay, user_w_lms, COLOR_OVERLAY_USER, display_w, display_h)

        # Combine panels into a single display image
        combined_display = np.hstack((panel_expert, panel_user, panel_overlay))
        cv2.imshow("Gym Form Analysis", combined_display)

        # Write frame to output video if VideoWriter is available
        if video_writer and video_writer.isOpened():
            video_writer.write(combined_display)

        # Control live view speed to match desired_frame_duration
        frame_processing_time = time.time() - frame_start_time
        wait_time_seconds = desired_frame_duration - frame_processing_time
        wait_time_ms = int(max(1, wait_time_seconds * 1000)) # Wait at least 1ms

        if cv2.waitKey(wait_time_ms) & 0xFF == ord('q'): # Exit if 'q' is pressed
            break

    print("Closing program.")
    pose.close() # Release MediaPipe Pose resources
    if video_writer and video_writer.isOpened():
        video_writer.release() # Release video writer
        print(f"Video saved to {OUTPUT_VIDEO_PATH}")
    cv2.destroyAllWindows() # Close all OpenCV windows

if __name__ == "__main__":
    # Check if video files exist before running main
    if not os.path.exists(EXPERT_VIDEO_PATH):
        print(f"Error: Expert video file not found at: {EXPERT_VIDEO_PATH}")
    elif not os.path.exists(USER_VIDEO_PATH):
        print(f"Error: User video file not found at: {USER_VIDEO_PATH}")
    else:
        main()
