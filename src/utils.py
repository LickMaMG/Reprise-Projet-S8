import os

class TackleWarnings:
	def __init__(self) -> None:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

