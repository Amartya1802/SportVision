from utils import (read_video, 
                   save_video)

from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2

def main():
    #reading video
    input_video_path = "input/tennisVid.mp4"
    video_frames = read_video(input_video_path)

    #detect players and ball
    player_tracker = PlayerTracker(model_path = 'yolov8x')
    ball_tracker = BallTracker(model_path = 'models/yolo5_last.pt')
    
    player_detections = player_tracker.detect_frames(video_frames, 
                                                     read_from_stub = True,
                                                     stub_path = "tracker_stubs/player_detections.pkl")
    
    ball_detections = ball_tracker.detect_frames(video_frames, 
                                                     read_from_stub = True,
                                                     stub_path = "tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    #court line detection
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    
    #draw output

    ## Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    #draw keypoints on court
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # draw frame number on top left corner
    frame_number = 0
    for frame in output_video_frames:
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_number += 1
    
    save_video(output_video_frames, "output/outputVid.avi")

if __name__ == "__main__":
    main()