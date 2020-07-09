class LoggerWriter:
    """LoggerWriter.

    see: https://stackoverflow.com/questions/19425736/
    how-to-redirect-stdout-and-stderr-to-logger-in-python
    """

    def __init__(self, printer):
        """__init__.

        Args:
            printer: Printer.
        """
        self.printer = printer

    def write(self, message):
        """write.

        Args:
            message: Message.
        """
        if message != '\n':
            self.printer(message)

    def flush(self):
        """flush."""
        pass
