# ============================================================
# AMAZON ML CHALLENGE 2026
# STAGE 3 — NORMALIZATION + NAME STRUCTURING
#          + ADDRESS STRUCTURING + COUNTRY HANDLING
#
# INPUT:
# Stage-2 Romanized TSV files
#
# IMPORTANT:
# Put your uploaded Stage-2 files in INPUT_DIR.
#
# Example:
# INPUT_DIR = "/content"
#
# If you upload the files directly into the current Colab
# working directory, leave:
#
# INPUT_DIR = ""
#
# OUTPUT:
# /content/stage3_normalized/
#
# CHUNK SIZE:
# 100,000 rows
#
# ============================================================


# ============================================================
# 1. INSTALL / IMPORT DEPENDENCIES
# ============================================================

%pip install -q jellyfish

import os
import gc
import re
import shutil
import unicodedata
from functools import lru_cache

import pandas as pd
import jellyfish

from tqdm.auto import tqdm


# ============================================================
# 2. DATASET PATH
# ============================================================
#
# IMPORTANT:
# Put the directory containing your uploaded Stage-2 files
# between the quotes below.
#
# Example:
# INPUT_DIR = "/content"
#
# Or:
# INPUT_DIR = "/content/my_dataset"
#
# If the files are directly in the current Colab directory:
# INPUT_DIR = ""
#
# ============================================================

INPUT_DIR = ""


# ============================================================
# 3. OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "/content/stage3_normalized"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 4. PROCESSING CONFIGURATION
# ============================================================

CHUNK_SIZE = 100_000


print("=" * 100)
print("AMAZON ML CHALLENGE 2026 — STAGE 3")
print("=" * 100)

print("\nInput directory:")
print(os.path.abspath(INPUT_DIR))

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nChunk size:")
print(f"{CHUNK_SIZE:,}")


# ============================================================
# 5. FILE DEFINITIONS
# ============================================================

INPUT_FILES = {

    "train_source1":
        "train_source1_stage2_romanized.tsv",

    "train_source2":
        "train_source2_stage2_romanized.tsv",

    "train_source3":
        "train_source3_stage2_romanized.tsv",

    "test_source1":
        "test_source1_stage2_romanized.tsv",

    "test_source2":
        "test_source2_stage2_romanized.tsv",

    "test_source3":
        "test_source3_stage2_romanized.tsv",

}


# ============================================================
# 6. CHECK INPUT FILES
# ============================================================

print("\n" + "=" * 100)
print("CHECKING STAGE-2 FILES")
print("=" * 100)

AVAILABLE_FILES = {}

for dataset_name, filename in INPUT_FILES.items():

    input_path = os.path.join(
        INPUT_DIR,
        filename
    )

    if os.path.exists(input_path):

        size_gb = (
            os.path.getsize(input_path)
            / (1024 ** 3)
        )

        AVAILABLE_FILES[
            dataset_name
        ] = input_path

        print(
            f"✓ {dataset_name:<20}"
            f"{size_gb:>8.2f} GB"
        )

    else:

        print(
            f"✗ {dataset_name:<20}"
            f"NOT FOUND"
        )


if not AVAILABLE_FILES:

    raise FileNotFoundError(
        "\nNo Stage-2 TSV files were found.\n"
        "Upload the files to Colab and set INPUT_DIR "
        "correctly."
    )


# ============================================================
# 7. NORMALIZATION HELPERS
# ============================================================


# ------------------------------------------------------------
# Unicode normalization
# ------------------------------------------------------------

def unicode_clean(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Unicode compatibility normalization
    text = unicodedata.normalize(
        "NFKC",
        text
    )

    # Remove control / invisible characters
    text = "".join(
        ch
        for ch in text
        if unicodedata.category(ch)
        not in {"Cc", "Cf"}
    )

    # Lowercase
    text = text.lower()

    return text


# ------------------------------------------------------------
# NAME-SPECIFIC NORMALIZATION
# ------------------------------------------------------------

LEGAL_SUFFIX_REPLACEMENTS = {

    # Private Limited
    "pvt": "private",
    "private": "private",

    # Limited
    "ltd": "limited",
    "limited": "limited",

    # Corporation
    "corp": "corporation",
    "corporation": "corporation",

    # Incorporated
    "inc": "incorporated",
    "incorporated": "incorporated",

    # Company
    "co": "company",
    "company": "company",

    # LLC
    "llc": "limited liability company",

    # LLP
    "llp": "limited liability partnership",

    # PLC
    "plc": "public limited company",
}


def normalize_name(text):

    """
    Name normalization.

    Performs:
    - Unicode normalization
    - lowercase
    - & → and
    - common slash equivalence
    - punctuation normalization
    - legal suffix normalization
    - whitespace normalization

    IMPORTANT:
    This does NOT use an LLM.
    This is deterministic preprocessing.
    """

    text = unicode_clean(text)

    if not text:
        return ""

    # --------------------------------------------------------
    # Handle special legal forms BEFORE general punctuation
    # --------------------------------------------------------

    # pvt/ltd → private limited
    text = re.sub(
        r"\bpvt\s*[/&-]\s*ltd\b",
        " private limited ",
        text
    )

    text = re.sub(
        r"\bprivate\s*[/&-]\s*limited\b",
        " private limited ",
        text
    )

    # l.l.c → llc
    text = re.sub(
        r"\bl\s*\.\s*l\s*\.\s*c\s*\.?\b",
        " llc ",
        text
    )

    # l.l.p → llp
    text = re.sub(
        r"\bl\s*\.\s*l\s*\.\s*p\s*\.?\b",
        " llp ",
        text
    )

    # --------------------------------------------------------
    # & / + → and
    # --------------------------------------------------------

    text = re.sub(
        r"\s*&\s*",
        " and ",
        text
    )

    text = re.sub(
        r"\s*\+\s*",
        " and ",
        text
    )

    # Slash between words
    text = re.sub(
        r"(?<=\w)\s*/\s*(?=\w)",
        " and ",
        text
    )

    # --------------------------------------------------------
    # Normalize punctuation to spaces
    # --------------------------------------------------------

    text = re.sub(
        r"[.,;:!?\"'`~^|\\()\[\]{}<>_=]+",
        " ",
        text
    )

    # Hyphen → space for name matching
    text = re.sub(
        r"[-]+",
        " ",
        text
    )

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    tokens = text.split()

    normalized_tokens = []

    for token in tokens:

        # Remove residual punctuation around tokens
        token = token.strip(
            ".,;:!?\"'`~^|\\()[]{}<>_=+-"
        )

        if not token:
            continue

        # Legal suffix replacement
        replacement = (
            LEGAL_SUFFIX_REPLACEMENTS
            .get(token, token)
        )

        normalized_tokens.append(
            replacement
        )

    # --------------------------------------------------------
    # Rebuild
    # --------------------------------------------------------

    text = " ".join(
        normalized_tokens
    )

    # --------------------------------------------------------
    # Final whitespace normalization
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ------------------------------------------------------------
# ADDRESS-SPECIFIC NORMALIZATION
# ------------------------------------------------------------

def normalize_address(text):

    """
    Address normalization.

    Keeps structural characters such as:
    - comma
    - slash
    - hyphen
    - #

    because they can carry address information.

    Performs:
    - Unicode normalization
    - lowercase
    - punctuation normalization
    - whitespace normalization

    Country is NOT modified here.
    """

    text = unicode_clean(text)

    if not text:
        return ""

    # Ampersand → and
    text = re.sub(
        r"\s*&\s*",
        " and ",
        text
    )

    # Normalize semicolon to comma
    text = re.sub(
        r"\s*;\s*",
        ", ",
        text
    )

    # Normalize repeated commas
    text = re.sub(
        r",+",
        ",",
        text
    )

    # Remove punctuation that is generally not useful
    text = re.sub(
        r"[.!?:\"'`~^|\\()\[\]{}<>_=+]+",
        " ",
        text
    )

    # Keep:
    #   /  address fractions
    #   -  building/unit numbers
    #   #  unit numbers
    #
    # Normalize their surrounding whitespace.
    text = re.sub(
        r"\s*/\s*",
        "/",
        text
    )

    text = re.sub(
        r"\s*-\s*",
        "-",
        text
    )

    text = re.sub(
        r"\s*#\s*",
        "#",
        text
    )

    # Repeated whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Normalize spaces around commas
    text = re.sub(
        r"\s*,\s*",
        ", ",
        text
    )

    # Remove repeated commas
    text = re.sub(
        r"(,\s*){2,}",
        ", ",
        text
    )

    return text.strip(" ,")


# ============================================================
# 8. NAME STRUCTURING
# ============================================================


def token_string(text):

    if not text:
        return ""

    tokens = text.split()

    return "|".join(tokens)


def compact_form(text):

    if not text:
        return ""

    # Keep Unicode alphanumeric characters.
    return "".join(
        ch
        for ch in text
        if ch.isalnum()
    )


def first_token(text):

    if not text:
        return ""

    tokens = text.split()

    return tokens[0] if tokens else ""


def last_token(text):

    if not text:
        return ""

    tokens = text.split()

    return tokens[-1] if tokens else ""


def token_count(text):

    if not text:
        return 0

    return len(
        text.split()
    )


def char_length(text):

    if not text:
        return 0

    return len(
        text.replace(" ", "")
    )


# ============================================================
# 9. PHONETIC REPRESENTATION
# ============================================================
#
# Phonetic representation is generated primarily from the
# NORMALIZED ROMAN name because phonetic algorithms such as
# Metaphone operate on Latin/Roman text.
#
# The original multilingual name is still preserved.
# ============================================================


@lru_cache(maxsize=200_000)
def phonetic_from_roman(text):

    if not text:
        return ""

    tokens = re.findall(
        r"[a-z]+",
        text
    )

    codes = []

    for token in tokens:

        try:

            code = jellyfish.metaphone(
                token
            )

            if code:
                codes.append(code)

        except Exception:

            continue

    return "|".join(codes)


# ============================================================
# 10. ADDRESS STRUCTURING HELPERS
# ============================================================


HOUSE_NUMBER_PATTERN = re.compile(
    r"""
    ^
    \s*
    (?:
        \#\s*
        |
        no\.?\s*
        |
        h\.?\s*no\.?\s*
        |
        house\s*
        |
        plot\s*
        |
        flat\s*
        |
        unit\s*
        |
        apt\.?\s*
        |
        apartment\s*
        |
        shop\s*
        |
        door\s*
    )?
    (
        [a-z]?\d+[a-z]?
        (?:
            [-/][a-z]?\d+[a-z]?
        )*
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE
)


NUMBER_PATTERN = re.compile(
    r"\b"
    r"[a-z]?\d+[a-z]?"
    r"(?:[-/][a-z]?\d+[a-z]?)?"
    r"\b",
    re.IGNORECASE
)


POSTAL_PATTERNS = [

    # US ZIP / ZIP+4
    re.compile(
        r"\b\d{5}(?:-\d{4})?\b"
    ),

    # India / other 6-digit postal codes
    re.compile(
        r"\b\d{6}\b"
    ),

    # Generic 4-6 digit fallback
    re.compile(
        r"\b\d{4,6}\b"
    ),

    # Generic alphanumeric postal format
    re.compile(
        r"\b[a-z]\d[a-z][ -]?\d[a-z]\d\b",
        re.IGNORECASE
    ),
]


STREET_TYPE_WORDS = {
    "street",
    "st",
    "road",
    "rd",
    "avenue",
    "ave",
    "boulevard",
    "blvd",
    "drive",
    "dr",
    "lane",
    "ln",
    "court",
    "ct",
    "place",
    "pl",
    "parkway",
    "pkwy",
    "highway",
    "hwy",
    "terrace",
    "ter",
    "way",
    "square",
    "sq",
}


def extract_house_number(address):

    if not address:
        return ""

    match = HOUSE_NUMBER_PATTERN.search(
        address
    )

    if not match:
        return ""

    return match.group(1)


def extract_postal_code(address, house_number=""):

    if not address:
        return ""

    candidates = []

    for pattern in POSTAL_PATTERNS:

        matches = pattern.findall(
            address
        )

        for match in matches:

            value = (
                match[0]
                if isinstance(match, tuple)
                else match
            )

            if not value:
                continue

            # Don't use house number as postal code
            if value == house_number:
                continue

            candidates.append(value)

    if not candidates:
        return ""

    # Last postal-like occurrence is usually nearest
    # the end of the address.
    return candidates[-1]


def split_address_parts(address):

    if not address:
        return []

    parts = [
        part.strip()
        for part in address.split(",")
        if part.strip()
    ]

    return parts


def has_street_type(text):

    if not text:
        return False

    words = set(
        re.findall(
            r"[a-z]+",
            text.lower()
        )
    )

    return bool(
        words.intersection(
            STREET_TYPE_WORDS
        )
    )


def is_house_only(part):

    if not part:
        return False

    return bool(
        HOUSE_NUMBER_PATTERN.fullmatch(
            part.strip()
        )
    )


def remove_house_prefix(
    part,
    house_number
):

    if not part:
        return ""

    if not house_number:
        return part.strip()

    # Remove the detected leading house/building number.
    result = re.sub(
        r"^\s*(?:#|no\.?|h\.?\s*no\.?|"
        r"house|plot|flat|unit|apt\.?|"
        r"apartment|shop|door)?\s*"
        r"[a-z]?\d+[a-z]?"
        r"(?:[-/][a-z]?\d+[a-z]?)?"
        r"\s*",
        "",
        part,
        count=1,
        flags=re.IGNORECASE
    )

    return result.strip(" ,-#")


def parse_address(
    normalized_address,
    country
):

    """
    Generic address parser.

    IMPORTANT:
    This is heuristic, not an external geocoder/database.

    It does not hard-code US / India / France.
    Country is used only to remove an exact trailing country
    token when the address itself redundantly contains it.
    """

    result = {

        "house_number": "",
        "postal_code": "",
        "street": "",
        "locality": "",
        "city": "",
        "state_region": "",
        "numeric_tokens": "",
        "address_tokens": "",
        "address_token_count": 0,
        "address_char_length": 0,
    }

    if not normalized_address:

        return result

    text = normalized_address.strip()

    result["address_tokens"] = (
        token_string(text)
    )

    result["address_token_count"] = (
        token_count(text)
    )

    result["address_char_length"] = (
        char_length(text)
    )

    # --------------------------------------------------------
    # House / building number
    # --------------------------------------------------------

    house_number = extract_house_number(
        text
    )

    result["house_number"] = house_number

    # --------------------------------------------------------
    # Postal / PIN / ZIP-like value
    # --------------------------------------------------------

    result["postal_code"] = (
        extract_postal_code(
            text,
            house_number
        )
    )

    # --------------------------------------------------------
    # Numeric tokens
    # --------------------------------------------------------

    numeric_values = []

    for match in NUMBER_PATTERN.findall(
        text
    ):

        if match not in numeric_values:
            numeric_values.append(match)

    result["numeric_tokens"] = (
        "|".join(numeric_values)
    )

    # --------------------------------------------------------
    # Split address into comma-separated segments
    # --------------------------------------------------------

    parts = split_address_parts(
        text
    )

    if not parts:

        return result

    # --------------------------------------------------------
    # Remove exact trailing country if duplicated
    #
    # IMPORTANT:
    # country is NOT normalized.
    # This is only an exact comparison.
    # --------------------------------------------------------

    country_text = (
        ""
        if pd.isna(country)
        else str(country).strip()
    )

    if (
        country_text
        and len(parts) > 1
        and parts[-1].casefold()
        == country_text.casefold()
    ):

        parts = parts[:-1]

    if not parts:

        return result

    # --------------------------------------------------------
    # Remove postal code from final segment for parsing
    # --------------------------------------------------------

    if result["postal_code"]:

        postal = result[
            "postal_code"
        ]

        parts = [
            re.sub(
                re.escape(postal),
                "",
                part,
                flags=re.IGNORECASE
            ).strip(" ,")
            for part in parts
        ]

        parts = [
            p
            for p in parts
            if p
        ]

    if not parts:

        return result

    # --------------------------------------------------------
    # Determine state/region and city heuristically
    # --------------------------------------------------------

    state_region = ""
    city = ""
    city_index = None
    state_index = None

    # Case 1:
    # Last segment is a short alphabetic code.
    #
    # Example:
    #   "san jose ca"
    #
    if len(parts) >= 2:

        trailing_code = re.search(
            r"\b([a-z]{2,3})$",
            parts[-1],
            flags=re.IGNORECASE
        )

        if (
            trailing_code
            and len(parts[-1].split()) >= 2
        ):

            code = trailing_code.group(1)

            before_code = (
                parts[-1][
                    :trailing_code.start()
                ]
                .strip()
            )

            if before_code:

                state_region = code
                city = before_code

                state_index = len(parts) - 1
                city_index = len(parts) - 1

    # --------------------------------------------------------
    # Case 2:
    # Last segment is state/region
    # --------------------------------------------------------

    if not state_region:

        if len(parts) >= 3:

            last = parts[-1]

            # Don't treat a numeric segment as state.
            if (
                not re.search(
                    r"\d",
                    last
                )
                and len(last) <= 50
            ):

                state_region = last
                state_index = len(parts) - 1

                # Generic city heuristic
                if len(parts) >= 5:

                    # For longer structures such as:
                    # house, locality, city, district, state
                    if is_house_only(
                        parts[0]
                    ):

                        city_index = len(parts) - 3

                    else:

                        city_index = len(parts) - 2

                else:

                    city_index = (
                        len(parts) - 2
                    )

                if (
                    city_index is not None
                    and 0 <= city_index < len(parts)
                ):

                    city = parts[
                        city_index
                    ]

        elif len(parts) == 2:

            city = parts[-1]
            city_index = 1

    # --------------------------------------------------------
    # Street extraction
    # --------------------------------------------------------

    street_index = None
    street = ""

    # Prefer a segment containing an explicit street type.
    for idx, part in enumerate(parts):

        if has_street_type(part):

            street_index = idx

            street = remove_house_prefix(
                part,
                house_number
            )

            break

    # If no street type exists, use the first segment
    # when it appears to contain house number + text.
    if (
        street_index is None
        and parts
    ):

        first_part = parts[0]

        if (
            house_number
            and not is_house_only(
                first_part
            )
        ):

            street_index = 0

            street = remove_house_prefix(
                first_part,
                house_number
            )

    # --------------------------------------------------------
    # Fallback street detection
    # --------------------------------------------------------

    if (
        street_index is None
        and len(parts) >= 3
    ):

        # If first segment is not just a house number,
        # it is often the street/address line.
        if not is_house_only(
            parts[0]
        ):

            street_index = 0

            street = remove_house_prefix(
                parts[0],
                house_number
            )

    # --------------------------------------------------------
    # Locality
    # --------------------------------------------------------

    locality_parts = []

    if city_index is not None:

        start_index = 0

        if street_index is not None:

            start_index = (
                street_index + 1
            )

        else:

            if (
                parts
                and is_house_only(parts[0])
            ):

                start_index = 1

        for idx in range(
            start_index,
            city_index
        ):

            if (
                idx != street_index
                and idx != state_index
            ):

                locality_parts.append(
                    parts[idx]
                )

    else:

        # Without a city, keep middle parts as locality.
        start_index = 1 if (
            parts
            and is_house_only(parts[0])
        ) else 0

        locality_parts = parts[
            start_index:
        ]

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    result["street"] = (
        street.strip()
    )

    result["locality"] = (
        ", ".join(
            locality_parts
        ).strip(" ,")
    )

    result["city"] = (
        city.strip()
    )

    result["state_region"] = (
        state_region.strip()
    )

    return result


# ============================================================
# 11. ADD STAGE 3 COLUMNS TO ONE CHUNK
# ============================================================

def process_chunk(df):

    # ========================================================
    # BASIC CHECK
    # ========================================================

    required_columns = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
        "business_name_basic",
        "business_address_basic",
        "business_name_roman",
        "business_address_roman",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + str(missing_columns)
        )


    # ========================================================
    # NAME NORMALIZATION
    # ========================================================

    tqdm.pandas(
        desc="Name original normalization",
        leave=False
    )

    df[
        "business_name_normalized_original"
    ] = (
        df[
            "business_name_basic"
        ]
        .progress_map(
            normalize_name
        )
    )


    tqdm.pandas(
        desc="Name Roman normalization",
        leave=False
    )

    df[
        "business_name_normalized_roman"
    ] = (
        df[
            "business_name_roman"
        ]
        .progress_map(
            normalize_name
        )
    )


    # ========================================================
    # ADDRESS NORMALIZATION
    # ========================================================

    tqdm.pandas(
        desc="Address original normalization",
        leave=False
    )

    df[
        "business_address_normalized_original"
    ] = (
        df[
            "business_address_basic"
        ]
        .progress_map(
            normalize_address
        )
    )


    tqdm.pandas(
        desc="Address Roman normalization",
        leave=False
    )

    df[
        "business_address_normalized_roman"
    ] = (
        df[
            "business_address_roman"
        ]
        .progress_map(
            normalize_address
        )
    )


    # ========================================================
    # NAME STRUCTURING — ORIGINAL
    # ========================================================

    original_name = df[
        "business_name_normalized_original"
    ]

    df[
        "business_name_original_tokens"
    ] = original_name.map(
        token_string
    )

    df[
        "business_name_original_compact"
    ] = original_name.map(
        compact_form
    )

    df[
        "business_name_original_first_token"
    ] = original_name.map(
        first_token
    )

    df[
        "business_name_original_last_token"
    ] = original_name.map(
        last_token
    )

    df[
        "business_name_original_token_count"
    ] = original_name.map(
        token_count
    )

    df[
        "business_name_original_char_length"
    ] = original_name.map(
        char_length
    )


    # ========================================================
    # NAME STRUCTURING — ROMAN
    # ========================================================

    roman_name = df[
        "business_name_normalized_roman"
    ]

    df[
        "business_name_roman_tokens"
    ] = roman_name.map(
        token_string
    )

    df[
        "business_name_roman_compact"
    ] = roman_name.map(
        compact_form
    )

    df[
        "business_name_roman_first_token"
    ] = roman_name.map(
        first_token
    )

    df[
        "business_name_roman_last_token"
    ] = roman_name.map(
        last_token
    )

    df[
        "business_name_roman_token_count"
    ] = roman_name.map(
        token_count
    )

    df[
        "business_name_roman_char_length"
    ] = roman_name.map(
        char_length
    )


    # ========================================================
    # PHONETIC — ROMAN FORM
    # ========================================================

    tqdm.pandas(
        desc="Name phonetic representation",
        leave=False
    )

    df[
        "business_name_roman_phonetic"
    ] = roman_name.progress_map(
        phonetic_from_roman
    )


    # ========================================================
    # ADDRESS STRUCTURING
    #
    # Use the normalized Roman address because it gives us
    # a common representation across multilingual records.
    #
    # The normalized original address remains preserved too.
    # ========================================================

    address_df = []

    for address, country in tqdm(
        zip(
            df[
                "business_address_normalized_roman"
            ],
            df["country"]
        ),
        total=len(df),
        desc="Address structuring",
        leave=False
    ):

        parsed = parse_address(
            address,
            country
        )

        address_df.append(
            parsed
        )


    parsed_address = pd.DataFrame(
        address_df,
        index=df.index
    )


    # Rename to final dataset columns
    parsed_address = parsed_address.rename(
        columns={
            "house_number":
                "address_house_number",

            "postal_code":
                "address_postal_code",

            "street":
                "address_street",

            "locality":
                "address_locality",

            "city":
                "address_city",

            "state_region":
                "address_state_region",

            "numeric_tokens":
                "address_numeric_tokens",

            "address_tokens":
                "address_tokens",

            "address_token_count":
                "address_token_count",

            "address_char_length":
                "address_char_length",
        }
    )


    for col in parsed_address.columns:

        df[col] = parsed_address[col]


    return df


# ============================================================
# 12. PROCESS ONE FILE IN CHUNKS
# ============================================================

def process_file(
    dataset_name,
    input_path,
    output_filename
):

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )


    print("\n" + "=" * 100)
    print(
        f"PROCESSING: {dataset_name}"
    )
    print("=" * 100)

    print(
        "Input :",
        input_path
    )

    print(
        "Output:",
        output_path
    )

    print(
        "Chunk :",
        f"{CHUNK_SIZE:,}"
    )


    # --------------------------------------------------------
    # Remove previous stage-3 output
    # --------------------------------------------------------

    if os.path.exists(
        output_path
    ):

        os.remove(
            output_path
        )


    first_chunk = True

    total_rows = 0

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
        # Stage 3 processing
        # ----------------------------------------------------

        df = process_chunk(
            df
        )


        # ----------------------------------------------------
        # Write chunk
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
            f"✓ Written: "
            f"{total_rows:,} rows"
        )


        # ----------------------------------------------------
        # Free memory
        # ----------------------------------------------------

        del df

        gc.collect()


    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print(
        f"\n✓ {dataset_name} COMPLETE"
    )

    print(
        f"Total rows: {total_rows:,}"
    )

    print(
        f"Saved to:\n{output_path}"
    )


# ============================================================
# 13. PROCESS ALL AVAILABLE DATASETS
# ============================================================

for dataset_name, input_path in (
    AVAILABLE_FILES.items()
):

    output_name = (
        f"{dataset_name}_stage3_normalized.tsv"
    )

    process_file(
        dataset_name,
        input_path,
        output_name
    )


# ============================================================
# 14. COPY GROUND TRUTH IF PRESENT
# ============================================================
#
# Ground truth is NOT normalized.
# It only contains Source-1 IDs and match labels.
#
# ============================================================

GROUND_TRUTH_CANDIDATES = [

    os.path.join(
        INPUT_DIR,
        "train_ground_truth.tsv"
    ),

    # Also support the original filename if uploaded.
    os.path.join(
        INPUT_DIR,
        "train_ground_truth.txt"
    )
]


for ground_truth_path in (
    GROUND_TRUTH_CANDIDATES
):

    if os.path.exists(
        ground_truth_path
    ):

        output_gt = os.path.join(
            OUTPUT_DIR,
            "train_ground_truth.tsv"
        )

        shutil.copy2(
            ground_truth_path,
            output_gt
        )

        print(
            "\n✓ Ground truth copied unchanged:"
        )

        print(
            output_gt
        )

        break


# ============================================================
# 15. DATASET QUALITY CHECK
# ============================================================

print("\n" + "=" * 100)
print("STAGE 3 OUTPUT CHECK")
print("=" * 100)


for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if not os.path.isfile(path):
        continue

    size_gb = (
        os.path.getsize(path)
        / (1024 ** 3)
    )

    print(
        f"{filename:<60}"
        f"{size_gb:>8.2f} GB"
    )


# ============================================================
# 16. SAMPLE INSPECTION
# ============================================================
#
# Read a small sample from every generated Stage-3 source
# file so you can manually verify the transformation.
# ============================================================

print("\n" + "=" * 100)
print("SAMPLE OUTPUT")
print("=" * 100)


for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    if (
        not filename.endswith(
            "_stage3_normalized.tsv"
        )
    ):
        continue


    path = os.path.join(
        OUTPUT_DIR,
        filename
    )


    sample = pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8",
        dtype="string",
        nrows=10
    )


    print(
        "\n" + "-" * 100
    )

    print(filename)

    print(
        "-" * 100
    )


    display(
        sample[
            [
                "entity_id",

                "business_name_basic",
                "business_name_roman",

                "business_name_normalized_original",
                "business_name_normalized_roman",

                "business_name_roman_compact",
                "business_name_roman_first_token",
                "business_name_roman_last_token",
                "business_name_roman_token_count",

                "business_name_roman_phonetic",

                "business_address_basic",
                "business_address_roman",

                "business_address_normalized_original",
                "business_address_normalized_roman",

                "address_house_number",
                "address_postal_code",
                "address_street",
                "address_locality",
                "address_city",
                "address_state_region",

                "country"
            ]
        ]
    )


# ============================================================
# 17. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 100)
print("STAGE 3 — NORMALIZATION + STRUCTURING COMPLETE")
print("=" * 100)

print("\nOutput directory:")
print(
    os.path.abspath(
        OUTPUT_DIR
    )
)

print("\nGenerated Stage-3 files:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    if filename.endswith(
        ".tsv"
    ):

        print(
            "  ",
            filename
        )


print(
    "\nCountry handling:"
)

print(
    "✓ Country column preserved exactly as supplied."
)

print(
    "✓ No US/India/France hard-coding."
)

print(
    "✓ Country exact-match will be created later "
    "when candidate pairs are formed."
)

print(
    "\n✓ Ready for Stage 4: blocking / candidate generation."
)
