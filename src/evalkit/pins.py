"""Every external version this project is measured against. Moved only at milestone gates (design §8.1)."""

HARNESS_REPO = "run-llama/ParseBench"
HARNESS_COMMIT = "6e4479ad9bda4347a81f7fe86dcf674071c352f9"  # main, after v2.2.1; first to carry ocria_1

DATA_REPO = "llamaindex/ParseBench-2"
DATA_REVISION = "2e95407575ba1ad6aa90842258482a561fae1b33"  # data_revision in leaderboard/results-v2.yaml

RESULTS_REPO = "llamaindex/ParseBench-2-results"
RESULTS_REVISION = "497e243f68cd861c80b5a1405e20758992d4a8de"  # results_revision of the OCRia-1 row
OCRIA_PIPELINE = "ocria_1"

OCRIA_REPO = "llamaindex/OCRia-1"
OCRIA_REVISION = "dae83aa51ecd0a8574d415d929a4d9e79348fd2a"

# OCRia-1's row in leaderboard-v2.csv at HARNESS_COMMIT (the numbers Task 4 must reproduce).
BOARD_OCRIA: dict[str, float] = {
    "overall": 85.24,
    "text": 87.87,
    "formatting": 91.58,
    "tables": 86.28,
    "charts": 77.66,
    "layout": 85.96,
    "forms": 82.10,
}
