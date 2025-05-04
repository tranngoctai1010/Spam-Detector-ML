from typing import Any

def validate_not_none(value: Any, name: str) -> None:
    """Kiểm tra xem giá trị có phải None không, nếu có thì báo lỗi."""
    if value is None:
        raise ValueError(f"{name} must not be None.")

def validate_non_empty_dict(value: dict, name: str) -> None:
    """Kiểm tra xem dictionary có rỗng không."""
    if not isinstance(value, dict) or len(value) == 0:
        raise ValueError(f"{name} must be a non-empty dictionary.")


def validate_isinstance(obj: Any, cls: Any) -> None:
    if not isinstance(obj, cls):
        raise TypeError(f"{obj} must be a {cls}.")