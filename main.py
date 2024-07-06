from utils import (read_video, 
                   save_video)

from trackers import PlayerTracker

def main():
    #reading video
    input_video_path = "input/tennisVid.mp4"
    video_frames = read_video(input_video_path)

    #detect video
    player_tracker = PlayerTracker(model_path = 'yolov8x')
    player_detections = player_tracker.detect_frames(video_frames)

    #draw output

    ## Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    save_video(output_video_frames, "output/outputVid.avi")

if __name__ == "__main__":
    main()