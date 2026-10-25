"""Domain errors -> HTTP status. Backends raise these; the API layer maps them to a JSON error body."""


class ServiceError(Exception):
    status_code = 500
    code = "internal_error"
    headers: dict | None = None


class Overloaded(ServiceError):
    """All LLM slots busy for longer than queue_timeout_s — tell the client when to retry."""
    status_code = 503
    code = "overloaded"

    def __init__(self, msg: str, retry_after_s: int):
        super().__init__(msg)
        self.headers = {"Retry-After": str(retry_after_s)}


class BadDocument(ServiceError):
    status_code = 422
    code = "bad_document"


class Unauthorized(ServiceError):
    status_code = 401
    code = "unauthorized"
