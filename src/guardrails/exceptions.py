class MaliciousDocumentError(Exception):
    """Raised when retrieved document content is unsafe."""

    def __init__(
        self,
        message: str = (
            "The request was blocked because "
            "a retrieved document was identified "
            "as potentially malicious."
        ),
    ):
        super().__init__(message)
