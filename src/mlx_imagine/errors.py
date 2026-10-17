"""The one error type the server raises for a request it refuses. The REST API and the MCP tools both turn it
into the error body of docs/api.md section 1.3."""

from typing import Any


class ApiError(Exception):
    def __init__(self, code: str, status: int, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


def invalid_request(message: str, **details: Any) -> ApiError:
    return ApiError("invalid_request", 400, message, details)


def not_found(message: str, **details: Any) -> ApiError:
    return ApiError("not_found", 404, message, details)


def path_not_allowed(message: str, **details: Any) -> ApiError:
    return ApiError("path_not_allowed", 403, message, details)


def conflict(code: str, message: str, **details: Any) -> ApiError:
    return ApiError(code, 409, message, details)


def unprocessable(code: str, message: str, **details: Any) -> ApiError:
    return ApiError(code, 422, message, details)


def not_supported(message: str, **details: Any) -> ApiError:
    return ApiError("not_supported", 501, message, details)
