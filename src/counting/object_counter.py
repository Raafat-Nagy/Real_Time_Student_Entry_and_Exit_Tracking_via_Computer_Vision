"""
Object Counting and Tracking Module

This module provides functionality for counting objects crossing defined regions in video streams.
It supports both line and polygon counting regions with directional counting (IN/OUT) capabilities.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Set
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from shapely.geometry import Point, LineString, Polygon
from ..utils import draw, CSVDataRecorder
from ..api.hall_status_api import report_hall_event

class Direction(Enum):
    """Enumeration for the direction of object movement.

    Attributes:
        IN: Represents movement into the defined area
        OUT: Represents movement out of the defined area
    """

    IN = "IN"
    OUT = "OUT"


class RegionType(Enum):
    """Enumeration for the type of counting region.

    Attributes:
        LINE: Straight line counting region
        POLYGON: Polygon-shaped counting region
    """

    LINE = "LINE"
    POLYGON = "POLYGON"


class Orientation(Enum):
    """Enumeration for the orientation of the counting region.

    Attributes:
        HORIZONTAL: Horizontally oriented region
        VERTICAL: Vertically oriented region
    """

    HORIZONTAL = "HORIZONTAL"
    VERTICAL = "VERTICAL"


@dataclass
class RegionInfo:
    """Container for counting region metadata and geometry.

    Attributes:
        region_type: Type of region (LINE or POLYGON)
        geometry: Shapely geometry object representing the region
        orientation: Primary orientation of the region (HORIZONTAL or VERTICAL)
    """

    region_type: RegionType
    geometry: LineString | Polygon
    orientation: Orientation


class ObjectCounter:
    """
    Main class for object counting and tracking across defined regions.

    Features:
    - Tracks object movement across lines or polygons
    - Counts IN/OUT directions based on region orientation
    - Maintains class-specific counts
    - Visualizes counting regions and object tracks
    - Optional CSV logging of counting events

    Usage Example:
        >>> counter = ObjectCounter(
        ...     region=[(100, 200), (300, 200)],
        ...     class_names={0: 'person', 1: 'car'},
        ...     csv_path='counts.csv'
        ... )
        >>> while processing_frames:
        ...     processed_frame = counter.process_frame(frame, boxes, track_ids, class_ids)
    """

    def __init__(
        self,
        region: List[Tuple[int, int]],
        class_names: Dict[int, str],
        csv_path: str | None = None,
        draw_boxes: bool = True,
        draw_tracking: bool = True,
        show_labels: bool = True,
        region_color: Tuple[int, int, int] = (200, 200, 0),
        line_thickness: int = 2,
        show_in_count: bool = True,
        show_out_count: bool = True,
        track_history_len: int = 25,
        crossing_threshold: float = 0.5,
        send_api_events: bool = True,
        hall_id: int = 1,
    ):
        """Initializes the ObjectCounter with configuration parameters.

        Args:
            region: List of (x,y) points defining counting region (min 2 points)
            class_names: Dictionary mapping class IDs to display names
            csv_path: Optional path for CSV output of counting events
            draw_boxes: Toggles bounding box drawing
            draw_tracking: Toggles movement trail drawing
            show_labels: Toggles display of object labels
            region_color: BGR color for region visualization
            line_thickness: Pixel width for drawn elements
            show_in_count: Toggles display of IN counts
            show_out_count: Toggles display of OUT counts
            track_history_len: Number of positions to retain for trail visualization
            crossing_threshold: Minimum movement distance to check for crossing
        """
        self._validate_inputs(region, class_names)

        self.region = np.array(region, dtype=np.int32)
        self.class_names = class_names
        self.csv_path = csv_path
        self.draw_tracking = draw_tracking
        self.draw_boxes = draw_boxes
        self.show_labels = show_labels
        self.region_color = region_color
        self.line_thickness = line_thickness
        self.show_in_count = show_in_count
        self.show_out_count = show_out_count
        self.track_history_len = max(2, track_history_len)
        self.crossing_threshold = crossing_threshold

        self.send_api_events = send_api_events
        self.hall_id = hall_id

        self.region_info = self._compute_region_info()
        print(__class__.__name__, vars(self))

        self.in_count_total = 0
        self.out_count_total = 0
        self.counted_ids: Set[int] = set()
        self.class_direction_counts = defaultdict(
            lambda: {Direction.IN.value: 0, Direction.OUT.value: 0}
        )
        self.object_histories = defaultdict(list)

        self.csv_logger = self._initialize_csv_logger() if self.csv_path else None
        self.csv_row_id = 1

    def _validate_inputs(
        self, points: List[Tuple[int, int]], names: Dict[int, str]
    ) -> None:
        """Validates constructor inputs meet minimum requirements.

        Raises:
            ValueError: If inputs fail validation checks
        """
        if len(points) < 2:
            raise ValueError("The 'region' list must contain at least 2 points.")
        if not names:
            raise ValueError("The 'class_names' dictionary cannot be empty.")
        for i, point in enumerate(points):
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                raise ValueError(
                    f"Point {i} in region must be a tuple/list of 2 coordinates."
                )

    def _initialize_csv_logger(self) -> CSVDataRecorder:
        """Configures CSV logging with standard headers.

        Returns:
            Configured CSVDataRecorder instance
        """
        headers = ["ID", "ClassName", "Direction", "Timestamp"]
        return CSVDataRecorder(self.csv_path, headers)

    def _compute_region_info(self) -> RegionInfo:
        """Analyzes region geometry to determine type and orientation.

        Returns:
            RegionInfo: Contains processed region characteristics
        """
        num_points = len(self.region)

        if num_points == 2:
            region_type = RegionType.LINE
            shape_geometry = LineString(self.region)
            width = abs(self.region[0][0] - self.region[1][0])
            height = abs(self.region[0][1] - self.region[1][1])
        else:
            region_type = RegionType.POLYGON
            shape_geometry = Polygon(self.region)
            x_coords, y_coords = self.region[:, 0], self.region[:, 1]
            width = x_coords.max() - x_coords.min()
            height = y_coords.max() - y_coords.min()

        orientation = Orientation.HORIZONTAL if width > height else Orientation.VERTICAL
        return RegionInfo(
            region_type=region_type, geometry=shape_geometry, orientation=orientation
        )

    def draw_region(self, image: np.ndarray) -> None:
        """Renders the counting region on the provided image.

        Args:
            image: The numpy array (frame) to draw on
        """
        is_polygon = self.region_info.region_type == RegionType.POLYGON
        cv2.polylines(
            image,
            [self.region],
            is_polygon,
            self.region_color,
            self.line_thickness,
        )

        for point in self.region:
            cv2.circle(image, tuple(point), 7, self.region_color, -1)
            cv2.circle(image, tuple(point), 8, (255, 255, 255), 1)

    def _is_crossing_region(
        self, current_pos: Tuple[int, int], prev_pos: Tuple[int, int]
    ) -> bool:
        """Determines if movement between positions crosses the region.

        Args:
            current_pos: (x,y) of current object position
            prev_pos: (x,y) of previous object position

        Returns:
            bool: True if crossing detected, False otherwise
        """
        if current_pos is None or prev_pos is None:
            return False

        movement_distance = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
        if movement_distance < self.crossing_threshold:
            return False

        if self.region_info.region_type == RegionType.LINE:
            movement_line = LineString([prev_pos, current_pos])
            return self.region_info.geometry.intersects(movement_line)
        else:
            polygon = self.region_info.geometry
            is_current_inside = polygon.contains(Point(current_pos))
            is_prev_inside = polygon.contains(Point(prev_pos))
            return is_current_inside != is_prev_inside

    def _determine_direction(
        self, current_pos: Tuple[int, int], prev_pos: Tuple[int, int]
    ) -> Direction:
        """Calculates movement direction relative to region orientation.

        Args:
            current_pos: Current object position
            prev_pos: Previous object position

        Returns:
            Direction: IN or OUT based on movement analysis
        """
        if self.region_info.orientation == Orientation.VERTICAL:
            return Direction.IN if current_pos[0] > prev_pos[0] else Direction.OUT
        else:
            return Direction.IN if current_pos[1] > prev_pos[1] else Direction.OUT

    def _process_object_count(
        self,
        current_pos: Tuple[int, int],
        prev_pos: Tuple[int, int] | None,
        track_id: int,
        class_id: int,
    ) -> None:
        """Handles counting logic for individual object movements.

        Args:
            current_pos: Current object position
            prev_pos: Previous object position (None if first detection)
            track_id: Unique identifier for the object
            class_id: Class identifier for the object
        """
        if prev_pos is None or track_id in self.counted_ids:
            return

        is_crossing = self._is_crossing_region(current_pos, prev_pos)

        if is_crossing:
            direction = self._determine_direction(current_pos, prev_pos)

            class_name = self.class_names[class_id]
            if direction == Direction.IN:
                self.in_count_total += 1
                self.class_direction_counts[class_name][Direction.IN.value] += 1
            else:
                self.out_count_total += 1
                self.class_direction_counts[class_name][Direction.OUT.value] += 1

            self.counted_ids.add(track_id)

            if self.send_api_events:
                report_hall_event(direction.value, self.hall_id)

            if self.csv_logger:
                self.csv_logger.add_row(
                    {
                        "ID": self.csv_row_id,
                        "ClassName": class_name,
                        "Direction": direction.value,
                        "Timestamp": cv2.getTickCount(),
                    }
                )
                self.csv_row_id += 1

    def _cleanup_inactive_tracks(self, current_track_ids: List[int]) -> None:
        """Removes tracking data for objects no longer detected.

        Args:
            current_track_ids: List of currently active track IDs
        """
        active_ids = set(current_track_ids)
        inactive_ids = set(self.object_histories.keys()) - active_ids

        for inactive_id in inactive_ids:
            del self.object_histories[inactive_id]

    def _reset_object_counted_status(
        self, track_id, current_pos: Tuple[int, int], prev_pos: Tuple[int, int]
    ) -> None:
        """Resets counting status for objects that have moved away from region.

        Args:
            track_id: ID of object to check
            current_pos: Current object position
            prev_pos: Previous object position
        """
        if track_id in self.counted_ids:
            if not self._is_crossing_region(current_pos, prev_pos):
                self.counted_ids.remove(track_id)

    def display_counts(self, image: np.ndarray) -> None:
        """Renders counting statistics on the output frame.

        Args:
            image: The frame to draw count information on
        """
        text_y_position = 30
        total_text = f"Total: (IN {self.in_count_total} | OUT {self.out_count_total})"
        draw.put_text_rect(image, total_text, (10, text_y_position), 0.7, 2)
        text_y_position += 40

        for class_name, counts in self.class_direction_counts.items():
            count_in = counts[Direction.IN.value] > 0
            count_out = counts[Direction.OUT.value] > 0

            if count_in or count_out:
                text_parts = [f"{class_name.capitalize()}:"]
                if self.show_in_count and count_in:
                    text_parts.append(f"IN {counts[Direction.IN.value]}")
                if self.show_out_count and count_out:
                    text_parts.append(f"OUT {counts[Direction.OUT.value]}")

                label_text = " ".join(text_parts)
                draw.put_text_rect(image, label_text, (10, text_y_position), 0.7, 1)
                text_y_position += 30

    def process_frame(
        self,
        image: np.ndarray,
        boxes: List[List[int]],
        track_ids: List[int],
        class_ids: List[int],
    ) -> np.ndarray:
        """Processes a single video frame for object counting.

        Args:
            image: Input video frame
            boxes: List of bounding boxes [x1,y1,x2,y2] for detected objects
            track_ids: List of unique IDs for each detected object
            class_ids: List of class IDs for each detected object

        Returns:
            np.ndarray: Processed frame with visualizations
        """
        self.draw_region(image)

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = map(int, box)
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

            if self.draw_boxes:
                label = self.class_names[class_id] if self.show_labels else None
                draw.draw_box_label(image, box, label)

            # Store tracking history
            history = self.object_histories[track_id]
            history.append(centroid)
            if len(history) > self.track_history_len:
                history.pop(0)

            prev_pos = (
                self.object_histories[track_id][-2]
                if len(self.object_histories[track_id]) > 1
                else None
            )

            self._process_object_count(centroid, prev_pos, track_id, class_id)

            if self.draw_tracking:
                draw.draw_centroid_and_tracks(image, history)

            self._reset_object_counted_status(track_id, centroid, prev_pos)

        self._cleanup_inactive_tracks(track_ids)
        self.display_counts(image)

        return image

    def get_counts(self) -> Dict:
        """Retrieves current counting statistics.

        Returns:
            Dictionary containing:
                - total_in: Total IN counts
                - total_out: Total OUT counts
                - classwise_counts: Per-class counting statistics
                - active_objects: Number of currently tracked objects
        """
        return {
            "total_in": self.in_count_total,
            "total_out": self.out_count_total,
            "classwise_counts": dict(self.class_direction_counts),
            "active_objects": len(self.object_histories),
        }

    def reset(self) -> None:
        """Resets all counting statistics and tracking data."""
        self.in_count_total = 0
        self.out_count_total = 0
        self.counted_ids.clear()
        self.class_direction_counts.clear()
        self.object_histories.clear()
        self.csv_row_id = 1
