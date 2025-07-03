import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Union


class CSVDataRecorder:
    """Manage data records by storing values and timestamps in a CSV file."""

    def __init__(
        self, file_path: str, headers: List[str], time_format: str = "%H:%M:%S"
    ):
        """
        Initialize the data recorder.

        Args:
            file_path: Path to the CSV file
            headers: Column headers (excluding time)
            time_format: Format for the time column (default: "%H:%M:%S")
        """
        self.file_path = self._create_dated_filepath(Path(file_path))
        self.headers = [*headers, "Time"]
        self.time_format = time_format
        self._ensure_header_exists()

    def _create_dated_filepath(self, base_path: Path) -> Path:
        """Generate filename with current date prefix."""
        date_str = datetime.now().strftime("%d-%m-%Y")
        return base_path.parent / f"{date_str}_{base_path.name}"

    def _ensure_header_exists(self) -> None:
        """Create file with headers if it doesn't exist or is empty."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists() or self.file_path.stat().st_size == 0:
            with self.file_path.open("w", newline="") as f:
                csv.writer(f).writerow(self.headers)

    def _prepare_row(self, record: Union[Dict[str, Any], List[Any]]) -> List[Any]:
        """
        Validate and format record with timestamp.

        Args:
            record: Data to record (dict or list)

        Returns:
            Formatted row with timestamp

        Raises:
            ValueError: If record format is invalid
        """
        if isinstance(record, dict):
            expected_keys = self.headers[:-1]
            if set(record.keys()) != set(expected_keys):
                raise ValueError(
                    f"Expected keys: {expected_keys}, got: {list(record.keys())}"
                )
            values = [record[key] for key in expected_keys]
        elif isinstance(record, list):
            if len(record) != len(self.headers) - 1:
                raise ValueError(
                    f"Expected {len(self.headers)-1} values, got {len(record)}"
                )
            values = record
        else:
            raise TypeError("Record must be a dict or list.")

        timestamp = datetime.now().strftime(self.time_format)
        return [*values, timestamp]

    def add_row(self, record: Union[Dict[str, Any], List[Any]]) -> None:
        """Add a single record to the CSV file."""
        row = self._prepare_row(record)
        self._write_rows([row])

    def add_rows(self, records: List[Union[Dict[str, Any], List[Any]]]) -> None:
        """Add multiple records to the CSV file."""
        rows = [self._prepare_row(r) for r in records]
        self._write_rows(rows)

    def _write_rows(self, rows: List[List[Any]]) -> None:
        """Write rows to CSV file."""
        with self.file_path.open("a", newline="") as f:
            csv.writer(f).writerows(rows)


if __name__ == "__main__":
    # Example usage
    recorder = CSVDataRecorder("students_attendance.csv", ["StudentID", "Name"])

    try:
        recorder.add_row({"StudentID": 1, "Name": "Raafta"})
        recorder.add_row([2, "Nagy"])
        recorder.add_rows(
            [{"StudentID": 3, "Name": "Mohamed"}, {"StudentID": 4, "Name": "Raafat"}]
        )
        recorder.add_rows([[5, "Raafat"], [6, "Nagy"]])
        print("Data recorded successfully")
    except Exception as e:
        print(f"Error: {e}")