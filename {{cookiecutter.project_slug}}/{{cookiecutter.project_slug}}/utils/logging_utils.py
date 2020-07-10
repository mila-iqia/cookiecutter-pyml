class LoggerWriter:  # pragma: no cover
    """LoggerWriter.

    see: https://stackoverflow.com/questions/19425736/
    how-to-redirect-stdout-and-stderr-to-logger-in-python
    """

    def __init__(self, printer):
        """__init__.

        Args:
            printer: (fn) function used to print message (e.g., logger.info).
        """
        self.printer = printer

    def write(self, message):
        """write.

        Args:
            message: (str) message to print.
        """
        if message != '\n':
            self.printer(message)

    def flush(self):
        """flush."""
        pass
