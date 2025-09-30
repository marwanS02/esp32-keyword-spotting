import argparse
import re
import sys
import logging
from pathlib import Path
from zipfile import ZipFile

logging.basicConfig()
logger = logging.getLogger("submit.py")

NUM_WORDS = 8
MODEL_NAME = "micro_kws_student"


def get_lines(file):
    with open(file, "r") as handle:
        lines = handle.readlines()
    stripped = [line.strip() for line in lines if len(line) > 0]
    return stripped


def count_lines(file):
    return len(get_lines(file))


def count_names(file):
    names = []
    for line in get_lines(file):
        assert "#" not in line, "# is not allowed in names.txt"
        names.append(line)
    return len(names)


def get_words(file):
    lines = get_lines(file)
    assert len(lines) == 1, "words.txt should only have a single line with comma separated values"
    line = lines[0]
    assert re.compile(
        r"^((yes|no|up|down|left|right|on|off),)+(yes|no|up|down|left|right|on|off)$"
    ).match(line), "Invalid format of words.txt"
    words = line.split(",")
    return words


def check(path, directory=True, allow_missing=False):
    if (directory and path.is_dir()) or (not directory and path.is_file()):
        return True
    if allow_missing:
        logger.warning("%s does not exist. Submission will be inclomplete!", path)
    else:
        logger.error(
            "%s does not exist. Use --ignore-missing to create an incomplete submission.", path
        )
        sys.exit(1)


def check_file(path, allow_missing=False):
    return check(path, directory=False, allow_missing=allow_missing)


def check_directory(path, allow_missing=False):
    return check(path, directory=True, allow_missing=allow_missing)


def gather_files(directory, allow_missing=False):
    files = []

    student = directory / "student"
    assert student.is_dir(), f"{student} directory does not exist"
    models = directory / "models"
    assert models.is_dir(), f"{models} directory does not exist"

    # Metadata
    names_path = student / "names.txt"
    assert names_path.is_file(), f"{names_path} file does not exist"
    assert count_names(names_path) > 0
    files.append(names_path)
    words_path = student / "words.txt"
    assert words_path.is_file(), f"{words_path} file does not exist"
    words = get_words(words_path)
    assert len(words) == NUM_WORDS, f"Expected {NUM_WORDS} in student/words.txt"
    files.append(words_path)

    # Code
    model_code_path = student / "model.py"
    if check_file(model_code_path, allow_missing=allow_missing):
        files.append(model_code_path)
    callbacks_code_path = student / "callbacks.py"
    if check_file(callbacks_code_path, allow_missing=allow_missing):
        files.append(callbacks_code_path)
    metrics_code_path = student / "metrics.py"
    if check_file(metrics_code_path, allow_missing=allow_missing):
        files.append(metrics_code_path)
    estimate_code_path = student / "estimate.py"
    if check_file(estimate_code_path, allow_missing=allow_missing):
        files.append(estimate_code_path)

    # Models
    words_str = "".join(words)
    model_path = models / f"{MODEL_NAME}_{words_str}.tflite"
    if check_file(model_path, allow_missing=allow_missing):
        files.append(model_path)
    model_quantized_path = models / f"{MODEL_NAME}_{words_str}_quantized.tflite"
    if check_file(model_quantized_path, allow_missing=allow_missing):
        files.append(model_quantized_path)

    # Accuracy
    # accuracy_path = directory / "accuracy.txt"
    # assert accuracy_path.is_file(), f"{accuracy_path} file does not exist"

    # Estimations
    # estimations_path = directory / "estimations.txt"
    # assert estimations_path.is_file(), f"{estimations_path} file does not exist"

    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="submission_1.zip")
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Will turn errors into warnings, allowing to create an incomplete submission.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print which files and directories are added to the submission.",
    )
    args = parser.parse_args()

    directory = Path(__file__).parent
    files = gather_files(directory, allow_missing=args.ignore_missing)

    print(f"Creating archive: {args.out}\n")

    with ZipFile(args.out, "w") as handle:
        for file in files:
            if args.verbose:
                print(f"Adding file: {file}")
            handle.write(file)
        print(f"\nDone. Please upload the {args.out} file to Moodle!")


if __name__ == "__main__":
    main()
