class ParseError(Exception):
    """The input bytes cannot be parsed (corrupt, encrypted, or not a PDF or image)."""
