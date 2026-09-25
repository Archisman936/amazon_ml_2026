# ============================================================
# AMAZON ML CHALLENGE 2026
# STAGE 2 — FULL DATASET
# SCRIPT DETECTION + INDIC → ROMAN
#
# CHUNK SIZE: 100,000 ROWS
#
# INPUT:
# /content/drive/MyDrive/Colab Notebooks/dataset/cleaned_stage1/
#
# OUTPUT:
# /content/drive/MyDrive/Colab Notebooks/dataset/stage2_romanized/
#
# IMPORTANT:
# - This starts Stage 2 again from the beginning.
# - Existing Stage-2 output files are removed first.
# - Processing is chunked so the entire dataset is NOT loaded
#   into RAM at once.
# - Original/basic-cleaned columns are preserved.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import gc
import re
import shutil
import unicodedata
from collections import Counter

import pandas as pd
from tqdm.auto import tqdm

from indic_transliteration import sanscript


# ============================================================
# 2. MOUNT GOOGLE DRIVE
# ============================================================

from google.colab import drive

drive.mount("/content/drive")


# ============================================================
# 3. PATHS
# ============================================================

BASE_DIR = (
    "/content/drive/MyDrive/"
    "Colab Notebooks/dataset"
)

# Stage 1 input
STAGE1_DIR = os.path.join(
    BASE_DIR,
    "cleaned_stage1"
)

# Stage 2 output
STAGE2_DIR = os.path.join(
    BASE_DIR,
    "stage2_romanized"
)

os.makedirs(
    STAGE2_DIR,
    exist_ok=True
)


# ============================================================
# 4. CONFIGURATION
# ============================================================

# 100,000 rows per chunk
CHUNK_SIZE = 100_000

print("=" * 100)
print("AMAZON ML CHALLENGE 2026 — STAGE 2")
print("=" * 100)

print("\nStage 1 input:")
print(STAGE1_DIR)

print("\nStage 2 output:")
print(STAGE2_DIR)

print("\nChunk size:")
print(f"{CHUNK_SIZE:,} rows")


# ============================================================
# 5. INPUT FILE DEFINITIONS
# ============================================================

TRAIN_FILES = {
    "train_source1": "train_source1_basic_cleaned.tsv",
    "train_source2": "train_source2_basic_cleaned.tsv",
    "train_source3": "train_source3_basic_cleaned.tsv",
}

TEST_FILES = {
    "test_source1": "test_source1_basic_cleaned.tsv",
    "test_source2": "test_source2_basic_cleaned.tsv",
    "test_source3": "test_source3_basic_cleaned.tsv",
}


# ============================================================
# 6. VERIFY STAGE-1 INPUT FILES
# ============================================================

print("\n" + "=" * 100)
print("CHECKING STAGE-1 INPUT FILES")
print("=" * 100)

AVAILABLE_TRAIN = {}
AVAILABLE_TEST = {}

for name, filename in TRAIN_FILES.items():

    path = os.path.join(
        STAGE1_DIR,
        filename
    )

    if os.path.exists(path):

        AVAILABLE_TRAIN[name] = path

        size_gb = (
            os.path.getsize(path)
            / (1024 ** 3)
        )

        print(
            f"✓ {name:<20} "
            f"{size_gb:.2f} GB"
        )

    else:

        print(
            f"✗ {name:<20} NOT FOUND"
        )


print()

for name, filename in TEST_FILES.items():

    path = os.path.join(
        STAGE1_DIR,
        filename
    )

    if os.path.exists(path):

        AVAILABLE_TEST[name] = path

        size_gb = (
            os.path.getsize(path)
            / (1024 ** 3)
        )

        print(
            f"✓ {name:<20} "
            f"{size_gb:.2f} GB"
        )

    else:

        print(
            f"✗ {name:<20} NOT FOUND"
        )


# ============================================================
# 7. DELETE OLD STAGE-2 OUTPUTS
# ============================================================
# Because you are retraining Stage 2 from scratch.
# ============================================================

print("\n" + "=" * 100)
print("RESETTING STAGE-2 OUTPUT DIRECTORY")
print("=" * 100)

for filename in os.listdir(STAGE2_DIR):

    path = os.path.join(
        STAGE2_DIR,
        filename
    )

    if os.path.isfile(path):

        os.remove(path)

    elif os.path.isdir(path):

        shutil.rmtree(path)

print("✓ Existing Stage-2 files removed")


# ============================================================
# 8. SCRIPT RANGES
# ============================================================

SCRIPT_RANGES = {

    "Devanagari": [
        (0x0900, 0x097F)
    ],

    "Bengali": [
        (0x0980, 0x09FF)
    ],

    "Gurmukhi": [
        (0x0A00, 0x0A7F)
    ],

    "Gujarati": [
        (0x0A80, 0x0AFF)
    ],

    "Oriya": [
        (0x0B00, 0x0B7F)
    ],

    "Tamil": [
        (0x0B80, 0x0BFF)
    ],

    "Telugu": [
        (0x0C00, 0x0C7F)
    ],

    "Kannada": [
        (0x0C80, 0x0CFF)
    ],

    "Malayalam": [
        (0x0D00, 0x0D7F)
    ],

    "Sinhala": [
        (0x0D80, 0x0DFF)
    ],
}


# ============================================================
# 9. SCRIPT → INDIC TRANSLITERATION SCHEME
# ============================================================

SCRIPT_TO_SCHEME = {

    "Devanagari":
        sanscript.DEVANAGARI,

    "Bengali":
        sanscript.BENGALI,

    "Gurmukhi":
        sanscript.GURMUKHI,

    "Gujarati":
        sanscript.GUJARATI,

    "Oriya":
        sanscript.ORIYA,

    "Tamil":
        sanscript.TAMIL,

    "Telugu":
        sanscript.TELUGU,

    "Kannada":
        sanscript.KANNADA,

    "Malayalam":
        sanscript.MALAYALAM,
}


# ============================================================
# 10. CHARACTER SCRIPT DETECTOR
# ============================================================

def get_char_script(ch):

    code = ord(ch)

    # --------------------------------------------------------
    # IMPORTANT:
    # Check Indic Unicode ranges FIRST.
    #
    # This keeps Indic vowel signs / marks attached to their
    # corresponding script.
    # --------------------------------------------------------

    for script, ranges in SCRIPT_RANGES.items():

        for start, end in ranges:

            if start <= code <= end:

                return script

    # --------------------------------------------------------
    # Unicode category
    # --------------------------------------------------------

    category = unicodedata.category(ch)

    # Generic combining mark
    if category.startswith("M"):

        return None

    # --------------------------------------------------------
    # Latin detection
    # --------------------------------------------------------

    try:

        unicode_name = unicodedata.name(
            ch,
            ""
        )

    except Exception:

        unicode_name = ""

    if "LATIN" in unicode_name:

        return "Latin"

    # --------------------------------------------------------
    # Ignore punctuation / numbers / symbols / controls
    # for script classification.
    # --------------------------------------------------------

    if (
        category.startswith("P")
        or category.startswith("N")
        or category.startswith("S")
        or category.startswith("C")
    ):

        return None

    return None


# ============================================================
# 11. FIELD-LEVEL SCRIPT DETECTOR
# ============================================================

def detect_script(text):

    if pd.isna(text):

        return "Empty"

    text = str(text)

    if not text.strip():

        return "Empty"

    counts = Counter()

    for ch in text:

        script = get_char_script(ch)

        if script is not None:

            counts[script] += 1

    if not counts:

        return "Other"

    # --------------------------------------------------------
    # Single script
    # --------------------------------------------------------

    if len(counts) == 1:

        return next(iter(counts))

    # --------------------------------------------------------
    # Multiple scripts
    # --------------------------------------------------------

    ordered = counts.most_common()

    dominant_script = ordered[0][0]
    dominant_count = ordered[0][1]

    second_count = ordered[1][1]

    total = sum(
        counts.values()
    )

    dominant_ratio = (
        dominant_count / total
    )

    # Ignore tiny contamination
    if second_count <= 2:

        return dominant_script

    if (
        dominant_ratio >= 0.90
        and second_count <= 5
    ):

        return dominant_script

    return "Mixed"


# ============================================================
# 12. TRANSLITERATION
# ============================================================

def transliterate_text(text):

    if pd.isna(text):

        return pd.NA

    text = str(text)

    if not text.strip():

        return text

    script = detect_script(text)

    # --------------------------------------------------------
    # LATIN → unchanged
    # --------------------------------------------------------

    if script == "Latin":

        return text

    # --------------------------------------------------------
    # PURE INDIC
    # --------------------------------------------------------

    if script in SCRIPT_TO_SCHEME:

        return sanscript.transliterate(
            text,
            SCRIPT_TO_SCHEME[script],
            sanscript.HK
        )

    # --------------------------------------------------------
    # MIXED SCRIPT
    # --------------------------------------------------------

    if script == "Mixed":

        output = []

        buffer = []

        current_script = None

        def flush():

            nonlocal buffer
            nonlocal current_script

            if not buffer:

                return

            chunk = "".join(buffer)

            # Indic chunk
            if current_script in SCRIPT_TO_SCHEME:

                chunk = sanscript.transliterate(
                    chunk,
                    SCRIPT_TO_SCHEME[current_script],
                    sanscript.HK
                )

            output.append(
                chunk
            )

            buffer = []

        for ch in text:

            ch_script = get_char_script(ch)

            # ------------------------------------------------
            # Indic character
            # ------------------------------------------------

            if ch_script in SCRIPT_TO_SCHEME:

                if current_script != ch_script:

                    flush()

                    current_script = ch_script

                buffer.append(ch)

            # ------------------------------------------------
            # Non-Indic character
            # ------------------------------------------------

            else:

                if current_script != ch_script:

                    flush()

                    current_script = ch_script

                buffer.append(ch)

        flush()

        return "".join(output)

    # --------------------------------------------------------
    # Unsupported / Other
    # --------------------------------------------------------

    return text


# ============================================================
# 13. ROMAN SEARCH FORM
# ============================================================

def make_roman_search(text):

    if pd.isna(text):

        return pd.NA

    text = str(text)

    # Lowercase
    text = text.lower()

    # Unicode normalize
    text = unicodedata.normalize(
        "NFKD",
        text
    )

    # Remove combining marks
    text = "".join(
        ch
        for ch in text
        if not unicodedata.combining(ch)
    )

    # Normalize whitespace
    text = " ".join(
        text.split()
    )

    return text


# ============================================================
# 14. PROCESS ONE FILE IN 100K CHUNKS
# ============================================================

def process_source_file(
    dataset_name,
    input_path,
    output_filename
):

    output_path = os.path.join(
        STAGE2_DIR,
        output_filename
    )

    print("\n" + "=" * 100)

    print(
        f"PROCESSING: {dataset_name}"
    )

    print("=" * 100)

    print(
        f"Input : {input_path}"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Chunk : {CHUNK_SIZE:,} rows"
    )

    print()

    # --------------------------------------------------------
    # Remove previous output if present
    # --------------------------------------------------------

    if os.path.exists(output_path):

        os.remove(output_path)

    first_chunk = True

    total_rows = 0

    name_script_counts = Counter()

    address_script_counts = Counter()

    # --------------------------------------------------------
    # Read file in chunks
    # --------------------------------------------------------

    reader = pd.read_csv(
        input_path,
        sep="\t",
        encoding="utf-8",
        dtype="string",
        chunksize=CHUNK_SIZE
    )

    for chunk_number, df in enumerate(
        reader,
        start=1
    ):

        print(
            f"\nChunk {chunk_number:,}"
            f" | rows = {len(df):,}"
        )

        # ----------------------------------------------------
        # Verify required columns
        # ----------------------------------------------------

        required_columns = [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "business_name_original",
            "business_name_basic",
            "business_address_original",
            "business_address_basic"
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                f"{dataset_name} is missing columns: "
                f"{missing_columns}"
            )

        # ----------------------------------------------------
        # BUSINESS NAME SCRIPT
        # ----------------------------------------------------

        name_scripts = []

        for value in tqdm(
            df["business_name_basic"],
            desc="Name script",
            unit="row"
        ):

            name_scripts.append(
                detect_script(value)
            )

        df["business_name_script"] = (
            name_scripts
        )

        # ----------------------------------------------------
        # ADDRESS SCRIPT
        # ----------------------------------------------------

        address_scripts = []

        for value in tqdm(
            df["business_address_basic"],
            desc="Address script",
            unit="row"
        ):

            address_scripts.append(
                detect_script(value)
            )

        df["business_address_script"] = (
            address_scripts
        )

        # ----------------------------------------------------
        # UPDATE SCRIPT DISTRIBUTIONS
        # ----------------------------------------------------

        name_script_counts.update(
            Counter(name_scripts)
        )

        address_script_counts.update(
            Counter(address_scripts)
        )

        # ----------------------------------------------------
        # BUSINESS NAME → ROMAN
        # ----------------------------------------------------

        roman_names = []

        for value in tqdm(
            df["business_name_basic"],
            desc="Name → Roman",
            unit="row"
        ):

            roman_names.append(
                transliterate_text(value)
            )

        df["business_name_roman"] = (
            roman_names
        )

        # ----------------------------------------------------
        # ROMAN NAME SEARCH FORM
        # ----------------------------------------------------

        df["business_name_roman_search"] = (
            df["business_name_roman"]
            .map(make_roman_search)
        )

        # ----------------------------------------------------
        # BUSINESS ADDRESS → ROMAN
        # ----------------------------------------------------

        roman_addresses = []

        for value in tqdm(
            df["business_address_basic"],
            desc="Address → Roman",
            unit="row"
        ):

            roman_addresses.append(
                transliterate_text(value)
            )

        df["business_address_roman"] = (
            roman_addresses
        )

        # ----------------------------------------------------
        # ROMAN ADDRESS SEARCH FORM
        # ----------------------------------------------------

        df["business_address_roman_search"] = (
            df["business_address_roman"]
            .map(make_roman_search)
        )

        # ----------------------------------------------------
        # WRITE CHUNK
        # ----------------------------------------------------

        df.to_csv(
            output_path,
            sep="\t",
            encoding="utf-8",
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk
        )

        first_chunk = False

        total_rows += len(df)

        print(
            f"✓ Written: {total_rows:,} rows"
        )

        # ----------------------------------------------------
        # FREE RAM
        # ----------------------------------------------------

        del df
        del name_scripts
        del address_scripts
        del roman_names
        del roman_addresses

        gc.collect()

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "-" * 100)

    print(
        f"✓ {dataset_name} COMPLETE"
    )

    print(
        f"Total rows: {total_rows:,}"
    )

    print("\nBusiness-name scripts:")

    for script, count in (
        name_script_counts
        .most_common()
    ):

        print(
            f"  {script:<15} "
            f"{count:,}"
        )

    print("\nBusiness-address scripts:")

    for script, count in (
        address_script_counts
        .most_common()
    ):

        print(
            f"  {script:<15} "
            f"{count:,}"
        )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )

    return output_path


# ============================================================
# 15. PROCESS TRAINING SOURCES
# ============================================================

print("\n" + "=" * 100)
print("TRAINING DATA")
print("=" * 100)

for name, input_path in AVAILABLE_TRAIN.items():

    process_source_file(
        dataset_name=name,
        input_path=input_path,
        output_filename=(
            f"{name}_stage2_romanized.tsv"
        )
    )


# ============================================================
# 16. COPY GROUND TRUTH
# ============================================================

print("\n" + "=" * 100)
print("GROUND TRUTH")
print("=" * 100)

# First look inside cleaned_stage1
ground_truth_candidates = [

    os.path.join(
        STAGE1_DIR,
        "train_ground_truth.tsv"
    ),

    # Fallback to original training directory
    os.path.join(
        BASE_DIR,
        "train",
        "train_ground_truth.tsv"
    )
]

ground_truth_input = None

for candidate in ground_truth_candidates:

    if os.path.exists(candidate):

        ground_truth_input = candidate

        break


if ground_truth_input is not None:

    ground_truth_output = os.path.join(
        STAGE2_DIR,
        "train_ground_truth.tsv"
    )

    shutil.copy2(
        ground_truth_input,
        ground_truth_output
    )

    print(
        "✓ Ground truth copied unchanged"
    )

    print(
        ground_truth_output
    )

else:

    print(
        "⚠ train_ground_truth.tsv was not found"
    )


# ============================================================
# 17. PROCESS TEST SOURCES
# ============================================================

print("\n" + "=" * 100)
print("TEST DATA")
print("=" * 100)

for name, input_path in AVAILABLE_TEST.items():

    process_source_file(
        dataset_name=name,
        input_path=input_path,
        output_filename=(
            f"{name}_stage2_romanized.tsv"
        )
    )


# ============================================================
# 18. FINAL OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("STAGE 2 COMPLETE")
print("=" * 100)

print("\nOutput directory:")
print(STAGE2_DIR)

print("\nGenerated files:")

total_output_size = 0

for filename in sorted(
    os.listdir(STAGE2_DIR)
):

    path = os.path.join(
        STAGE2_DIR,
        filename
    )

    if os.path.isfile(path):

        size_gb = (
            os.path.getsize(path)
            / (1024 ** 3)
        )

        total_output_size += size_gb

        print(
            f"  {filename:<55}"
            f"{size_gb:>8.2f} GB"
        )

print(
    f"\nTotal Stage-2 output size: "
    f"{total_output_size:.2f} GB"
)

print(
    "\n✓ Stage 2 finished successfully."
)
