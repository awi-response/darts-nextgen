"""A singleton class to manage rich progress bars for the application."""

import rich.console

#  from rich.progress import Progress


class RichManagerSingleton:
    """A singleton class to manage rich progress bars for the application."""

    _instance = None

    def __new__(cls):
        """Create a new instance of the RichProgressManager if it does not exist yet."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self):
        """Initialize the RichProgressManager."""
        self.console = rich.console.Console()
        # self.progress = Progress(console=self.console).__enter__()

    def __del__(self):
        """Exit the RichProgressManager."""
        # try:
        #     self.progress.__exit__(None, None, None)
        # except ImportError as e:
        #     if e.msg == "sys.meta_path is None, Python is likely shutting down":
        #         pass
        #     else:
        #         raise e


RichManager = RichManagerSingleton()
