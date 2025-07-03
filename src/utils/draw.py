import cv2
import numpy as np


def put_text_rect( 
    image,
    text: str,
    position,
    scale: float = 0.7,
    thickness: int = 1,
    text_color=(0, 0, 0),
    rect_color=(255, 255, 255),
    padding: int = 10,
):
    """
    Draws text with a background rectangle on the given image.

    Args:
        image (numpy.ndarray): The image on which to draw the text and rectangle.
        text (str): The text to display.
        position (tuple): The (x, y) coordinates of the bottom-left corner of the text.
        scale (float): Font scale factor.
        thickness (int): Thickness of the text lines.
        text_color (tuple): Color of the text in BGR format.
        rect_color (tuple): Color of the background rectangle in BGR format.
        padding (int): Padding around the text within the rectangle.

    Returns:
        None
    """
    x, y = position

    # Get the size of the text
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )

    # Calculate the coordinates of the rectangle
    rect_top_left = (x - padding, y + padding)
    rect_bottom_right = (x + text_width + padding, y - text_height - padding)

    # Draw the background rectangle
    cv2.rectangle(image, rect_top_left, rect_bottom_right, rect_color, cv2.FILLED)

    # Draw the text
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        text_color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_box_label(
    image,
    box,
    label: str | None = None,
    scale: float = 0.7,
    thickness: int = 1,
    text_color=(0, 0, 0),
    rect_color=(255, 255, 255),
    padding=5,
    box_color=(255, 255, 255),
    box_thickness=2,
):
    """
    Draws a bounding box with a label on the given image.

    Args:
        image (numpy.ndarray): The image on which to draw the box and label.
        box (tuple): The bounding box coordinates in xyxy format (x1, y1, x2, y2).
        label (str): The label text to display.
        scale (float): Font scale factor.
        thickness (int): Thickness of the text lines.
        text_color (tuple): Color of the text in BGR format.
        rect_color (tuple): Color of the background rectangle for the label in BGR format.
        padding (int): Padding around the text within the rectangle.
        box_color (tuple): Color of the bounding box in BGR format.
        box_thickness (int): Thickness of the bounding box lines.

    Returns:
        None
    """
    x1, y1, x2, y2 = map(int, box)

    # Draw the bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, box_thickness)

    if label is not None:
        # Get the size of the text
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )

        # Calculate the position for the label background rectangle
        rect_top_left = (x1, y1 - text_height - 2 * padding)
        rect_bottom_right = (x1 + text_width + 2 * padding, y1)

        # Draw the background rectangle for the label
        cv2.rectangle(image, rect_top_left, rect_bottom_right, rect_color, cv2.FILLED)

        # Draw the label text
        cv2.putText(
            image,
            label.capitalize(),
            (x1 + padding, y1 - padding),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA,
        )


def draw_centroid_and_tracks(
    img,
    track: list,
    max_track: int = 20,
    color: tuple = (255, 255, 255),
    track_thickness: int = 2,
):
    """
    Draw centroid point and track trails.

    Args:
        img (Image.Image or numpy array): The image to annotate.
        track (list): object tracking points for trails display
        color (tuple): tracks line color
        track_thickness (int): track line thickness value
    """
    if len(track) >= max_track:
        track = track[(-max_track - 1) : -1]
    points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [points], isClosed=False, color=color, thickness=track_thickness)
    cv2.circle(
        img, (int(track[-1][0]), int(track[-1][1])), track_thickness * 2, color, -1
    )
