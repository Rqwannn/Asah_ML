from typing import Optional, Dict, Any


def json_response(status: str, code: int, extended_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Success response generator.

    :param status: Action status like CREATE_ACTION, READ_ACTION, etc.
    :param extended_data: Optional additional data
    :return: Dictionary response
    """
    data = {
        "STATUS": status,
        "CODE": code,
    }

    if extended_data is not None:
        data["DATA"] = extended_data

    return data


def fail_response(status: str, code: int, err: Optional[str] = "") -> Dict[str, Any]:
    """
    Failure response generator.

    :param status: Status string
    :param err: Optional error message
    :return: Dictionary response
    """
    data = {
        "STATUS": status,
        "CODE": code,
    }

    if err:
        data["ERROR"] = err

    return data
