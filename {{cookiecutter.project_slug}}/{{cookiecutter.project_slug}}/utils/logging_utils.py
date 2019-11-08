class LoggerWriter:
    # see: https://stackoverflow.com/questions/19425736/
    # how-to-redirect-stdout-and-stderr-to-logger-in-python
    def __init__(self, printer):
        self.printer = printer

    def write(self, message):
        if message != '\n':
            self.printer(message)

    def flush(self):
        pass
