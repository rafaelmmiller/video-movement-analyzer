# 🎥👀 Video Movement Analyzer

A simple python tool for detecting movement in MP4 camera surveillance videos. This was made for a specific use case, but it can be used for other purposes. Feel free to use it as you see fit 😃.

## Features

- Automated movement detection
- Configurable sensitivity
- Region-based filtering
- Frame sequence analysis
- Saves detected frames
- Generates analyzed videos

## Installation

1. Clone the repository
```bash
git clone https://github.com/rafaelmmiller/video-movement-analyzer.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Add .mp4 videos to the `input/` directory

4. Run the analyzer:
```bash
python video_analyzer.py
```

## Project Structure
```
video-movement-analyzer/
├── input/ # Place input videos here
├── output/
│ ├── detection_frames/ # Detected movement frames
│ └── _analyzed.mp4 # Analyzed videos
├── video_analyzer.py # Main script
├── requirements.txt # Python dependencies
└── README.md
```

## Usage

1. Place your video files (MP4 format) in the `input/` directory

2. Run the analyzer:
```bash
python video_analyzer.py
```


3. Check results in the `output/` directory:
   - `detection_frames/`: Contains frames where movement was detected
   - `*_analyzed.mp4`: Videos with movement detection visualization

## Configuration

You can adjust detection parameters in `video_analyzer.py` (line 22):

```python
# Core detection parameters
self.min_area = 1000 # Minimum area for movement detection
self.min_width = 30 # Minimum width of movement
self.min_height = 30 # Minimum height of movement
self.frame_skip = 3 # Process every Nth frame
self.learning_rate = 0.005 # Background adaptation rate
self.track_frames = 10 # Number of frames to track movement
```

### Ignore Regions

You can define regions to ignore by uncommenting and modifying the `ignore_region` in `main()` (line 239). This will draw a blue border around the region:

```python
ignore_region = [
(0, height/2), # top-left
(width/2, height/2 - 120), # top-center
(width, height/2), # top-right
(width, height), # bottom-right
(0, height) # bottom-left
]
```

## Output
The outputs contain green rectangles for detected movement and blue borders for ignored regions (if set).

The analyzer outputs:
1. JPG frames with detected movement
2. Analyzed video with detected movement

## Visualization

To see real-time detection, uncomment these lines in `analyze_video()`:

```python
cv2.imshow('Video Analysis', frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
  break
```

This will open a window to display the video with detected movement. Press 'q' to close the window.

## TODO:

- [ ] Streams videos from camera
- [ ] Send data to S3/Glacier
- [ ] Add metadata info so we can jump into the point that movement was detected
- [ ] UI

## Contributing

Contributions are welcome! Please follow the regular GitHub open-source contributions process: fork the repository, make a pull request (PR), and describe the feature or bugfix in detail.

## License

This project is open-sourced under the MIT License.
