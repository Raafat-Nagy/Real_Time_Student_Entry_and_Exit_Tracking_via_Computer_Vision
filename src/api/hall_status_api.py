import requests
from typing import Literal


def report_hall_event(status: Literal["IN", "OUT"], hall_id: int = 1):
    """
    Reports a hall event (entry or exit) to the remote API.

    Args:
        status (Literal["IN", "OUT"]): The event status, must be either "IN" or "OUT".
        hall_id (int): The ID of the hall to include in the API call.

    Behavior:
        - Sends an HTTP GET request to the backend API to log entry or exit.
        - If status is invalid, prints a warning and exits.
        - On success, prints the API response JSON.
        - On failure, prints the status code and error text.

    Example:
        report_hall_event("IN", hall_id=3)
    """
    if status == "IN":
        url = f"https://nextgenedu-database.azurewebsites.net/api/hall/enter/{hall_id}"
    elif status == "OUT":
        url = f"https://nextgenedu-database.azurewebsites.net/api/hall/exit/{hall_id}"
    else:
        raise ValueError("Invalid status. Use 'IN' or 'OUT'.")

    headers = {
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print("Error:", response.status_code, response.text)
