"""Domain errors -> HTTP status. Backends raise these; the API layer turns them into ``{error, detail, request_id}``."""


class ServiceError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, detail: str, code: str | None = None, headers: dict | None = None):
        super().__init__(detail)
        self.code = code or self.code
        self.headers = headers


class NotFound(ServiceError):          # unknown_ticker, memo_not_found
    status_code = 404
    code = "not_found"


class Unprocessable(ServiceError):     # insufficient_data
    status_code = 422
    code = "unprocessable"


class Unavailable(ServiceError):       # llm_unavailable, sentiment_unavailable, overloaded
    status_code = 503
    code = "unavailable"
