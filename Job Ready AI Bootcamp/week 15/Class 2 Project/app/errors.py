"""Domain errors — backend (real বা fake) এগুলো raise করে, API layer HTTP status-এ map করে।

HIGHLIGHTS: Week 12-এর exception (OllamaConnectionError, ExtractionError) সরাসরি
route-এ import করলে tests-এও Week 12 import করতে হত। তাই নিজস্ব ছোট exception
hierarchy রাখা হয়েছে; real adapter অনুবাদ করে দেয়।
"""


class ServiceError(Exception):
    status_code = 500
    code = "internal_error"


class LLMUnavailable(ServiceError):
    status_code = 503
    code = "llm_unavailable"


class BadDocument(ServiceError):
    status_code = 422
    code = "bad_document"
