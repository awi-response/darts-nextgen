class DartsAcquisitionError(Exception):  # noqa: D100, D101
    def __init__(self, *args):  # noqa: D107
        super().__init__(*args)
