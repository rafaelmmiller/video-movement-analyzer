"""
Video Movement Analyzer
A tool for detecting and analyzing movement in videos using computer vision.

Features:
- Movement detection with configurable sensitivity
- Region-based movement filtering
- Frame sequence analysis
- Multi-video batch processing
"""

import cv2
import numpy as np
import os


class VideoAnalyzer:
    def __init__(self, video_path):
        """Initialize video analyzer with detection parameters."""
        self.video_path = video_path
        
        # Core detection parameters
        self.min_area = 1000
        self.min_width = 30
        self.min_height = 30
        self.frame_skip = 3
        self.learning_rate = 0.005
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=32,
            detectShadows=False
        )
        
        # Minimum dimensions for detection
        self.min_width = 30
        self.min_height = 30
        
        # Create output directory
        self.detection_frames_dir = './output/detection_frames'
        os.makedirs(self.detection_frames_dir, exist_ok=True)
        
        self.ignore_mask = None
        
        # Add tracking parameters
        self.track_frames = 10  # Number of frames to track after detection
        self.active_regions = []  # List of regions being tracked

    def get_video_dimensions(self):
        """Get the dimensions of the video frames."""
        cap = cv2.VideoCapture(self.video_path)
        _, frame = cap.read()
        cap.release()
        return frame.shape[:2]
        
    def set_ignore_region(self, points):
        """Set a region to ignore in movement detection."""
        if not points:
            self.ignore_mask = None
            return
            
        cap = cv2.VideoCapture(self.video_path)
        _, frame = cap.read()
        cap.release()
        
        height, width = frame.shape[:2]
        self.ignore_mask = np.ones((height, width), dtype=np.uint8)
        
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(self.ignore_mask, [points_array], 0)
        self.ignore_region_points = points_array

    def track_region(self, frame, region):
        """Analyze movement within a specific region"""
        x, y, w, h = region
        roi = frame[y:y+h, x:x+w]
        
        # Apply same processing to ROI
        blur = cv2.GaussianBlur(roi, (31, 31), 0)
        fg_mask = self.bg_subtractor.apply(blur, learningRate=self.learning_rate)
        
        # Calculate movement percentage in ROI
        movement_pixels = np.count_nonzero(fg_mask)
        total_pixels = fg_mask.size
        movement_percentage = (movement_pixels / total_pixels) * 100
        
        return movement_percentage > 15  # Increased from 5% to 15%
        
    def analyze_video(self, output_path=None):
        cap = cv2.VideoCapture(self.video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame range for middle section
        start_frame = int(total_frames * 0)
        end_frame = int(total_frames * 1)
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        
        # Initialize video writer
        if output_path:
            out = cv2.VideoWriter(
                output_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                fps, 
                (frame_width, frame_height)
            )

        # Track frames with movement for batch saving
        frame_buffer = []  # Store last track_frames frames with their detections
        detections_log = []
        frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue
            
            # Process movement in current frame
            has_movement = False
            current_detection = None
            
            # Apply ignore mask if it exists
            if self.ignore_mask is not None:
                # Draw blue border around ignored region
                cv2.polylines(frame, [self.ignore_region_points], True, (255, 0, 0), 2)
                
                # Apply mask to frame before processing
                blur = cv2.GaussianBlur(frame, (31, 31), 0)
                fg_mask = self.bg_subtractor.apply(blur, learningRate=self.learning_rate)
                fg_mask = fg_mask * self.ignore_mask
            else:
                blur = cv2.GaussianBlur(frame, (31, 31), 0)
                fg_mask = self.bg_subtractor.apply(blur, learningRate=self.learning_rate)
            
            # Clean up mask and find contours
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
            
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                
                if (area < self.min_area or 
                    w < self.min_width or 
                    h < self.min_height or
                    w * h > frame_width * frame_height * 0.5):
                    continue
                
                has_movement = True
                current_detection = (x, y, w, h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green rectangle for detection
                
                detections_log.append({
                    'frame': frame_count,
                    'time': frame_count/fps,
                    'area': area,
                    'bbox': (x, y, w, h)
                })
            
            # Handle frame buffer for sequence detection
            frame_buffer.append((frame_count, frame.copy(), has_movement, current_detection))
            
            if len(frame_buffer) > self.track_frames:
                # Check if the oldest frame has movement
                oldest_frame_count, oldest_frame, had_movement, old_detection = frame_buffer[0]
                
                # Check if there is future movement
                future_movement = any(f[2] for f in frame_buffer[1:])
                
                # If there was movement and future movement, save the frame
                if had_movement and future_movement and old_detection:
                    x, y, w, h = old_detection
                    cv2.rectangle(oldest_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if self.ignore_mask is not None:
                        cv2.polylines(oldest_frame, [self.ignore_region_points], True, (255, 0, 0), 2)
                    
                    # Save the frame
                    frame_filename = f'motion_frame_{oldest_frame_count:04d}.jpg'
                    frame_path = os.path.join(self.detection_frames_dir, frame_filename)
                    cv2.imwrite(frame_path, oldest_frame)
                
                frame_buffer.pop(0)
            
            # Display frame, uncomment to see the frame
            # cv2.imshow('Video Analysis', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            if output_path:
                out.write(frame)
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        return detections_log


def main():
    """Process all videos in the input directory."""
    input_dir = './input'
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    for video_file in os.listdir(input_dir):
        if not video_file.endswith('.mp4'):
            continue
            
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(input_dir, video_file)
        output_path = os.path.join(output_dir, f'{video_name}_analyzed.mp4')
        
        detection_frames_dir = os.path.join('./output/detection_frames', video_name)
        os.makedirs(detection_frames_dir, exist_ok=True)
        
        print(f"\nProcessing video: {video_file}")
        
        analyzer = VideoAnalyzer(video_path)
        analyzer.detection_frames_dir = detection_frames_dir  # Set specific directory for this video
        
        height, width = analyzer.get_video_dimensions()
        
        # Define region to ignore
        ignore_region = []
        # ignore_region = [
        #     (0, height/2),                    # top-left
        #     (width/2, height/2 - 120),  # top-center
        #     (width, height/2),                # top-right
        #     (width, height),                  # bottom-right
        #     (0, height)                       # bottom-left
        # ]
        analyzer.set_ignore_region(ignore_region)
        
        detections = analyzer.analyze_video(output_path=output_path)
        
        print(f"Results for {video_file}:")
        print(f"Total movements detected: {len(detections)}")
        
        if detections:
            print("Largest movement areas:")
            sorted_detections = sorted(detections, key=lambda x: x['area'], reverse=True)
            for i, detection in enumerate(sorted_detections[:5], 1):
                print(f"{i}. Frame {detection['frame']}: Area = {detection['area']:.2f} pixels")


if __name__ == "__main__":
    main() 