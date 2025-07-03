import cv2
from ultralytics import YOLO
from src.counting import ObjectCounter


def process_video(
    model_path: str,
    csv_path: str,
    input_video_path: str,
    region: list,
    draw_boxes: bool = True,
    draw_tracking: bool = True,
    show_labels: bool = True,
    send_api_events: bool = True,
    hall_id: int = 1,
    verbose: bool = True,
):
    model = YOLO(model_path)

    # Create ObjectCounter instance
    counter = ObjectCounter(
        region=region,
        class_names=model.names,
        csv_path=csv_path,
        draw_boxes=draw_boxes,
        draw_tracking=draw_tracking,
        show_labels=show_labels,
        send_api_events=send_api_events,
        hall_id=hall_id,
    )

    cap = cv2.VideoCapture(input_video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model on the frame
        results = model.track(frame, persist=True, verbose=verbose,classes=[0])

        if results[0].boxes.id is None:
            continue

        # Extract bounding boxes, track IDs, and class indices
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int).tolist()
        clss = results[0].boxes.cls.cpu().numpy().astype(int).tolist()

        processed_frame = counter.process_frame(frame, boxes, track_ids, clss)

        # Display the processed frame
        cv2.imshow("Processed Frame", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(counter.get_counts())


if __name__ == "__main__":
    model_path = "./models/VisDrone_YOLOv8s_V2.pt"
    csv_path = "./data/outputs/record.csv"
    input_video_path = "./data/inputs/people_in_marathon.mp4"

    # Define the counting region (line or polygon)
    # region = [(25, 470), (25, 500), (1260, 500), (1260, 470)]
    # region = [(25, 500), (1260, 500)]

    region = [(25, 590), (1260, 590)]

    process_video(
        model_path=model_path,
        csv_path=csv_path,
        input_video_path=input_video_path,
        region=region,
        draw_boxes=True,
        draw_tracking=True,
        show_labels=False,
        send_api_events=False,
        verbose=False,
    )
