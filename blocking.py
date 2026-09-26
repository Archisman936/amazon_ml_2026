# ============================================================
# AMAZON ML CHALLENGE 2026
# STAGE 4 — BLOCKING / CANDIDATE GENERATION
# ============================================================
#
# IMPORTANT:
# - Uses ACTUAL Stage-3 column names
# - Train / Test can be in different directories
# - No expected row-count checks
# - No dataset completeness warnings
# - Processes whatever rows are present
# - Uses DuckDB for scalable blocking
# - Generates train + test candidates
# - Evaluates training blocking recall
# - Creates final candidate_pairs.tsv
#
# ACTUAL STAGE-3 NAME COLUMNS:
#   business_name_normalized_original
#   business_name_normalized_roman
#   business_name_roman_phonetic
#
# ACTUAL STAGE-3 ADDRESS COLUMNS:
#   address_house_number
#   address_postal_code
#   address_locality
#   address_city
#   address_state_region
#
# ============================================================


# ============================================================
# 1. INSTALL DEPENDENCIES
# ============================================================

!pip -q install duckdb scikit-learn pandas numpy pyarrow jellyfish


# ============================================================
# 2. IMPORTS
# ============================================================

import os
import gc
import json
import time
import warnings

import duckdb
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


# ============================================================
# 3. CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# TRAIN STAGE-3 DIRECTORY
# ------------------------------------------------------------

TRAIN_STAGE3_DIR = (
    "/content/drive/MyDrive/stage3_normalized"
)


# ------------------------------------------------------------
# TEST STAGE-3 DIRECTORY
# ------------------------------------------------------------

TEST_STAGE3_DIR = (
    "/content/drive/MyDrive/Colab Notebooks/dataset/stage3_normalized"
)


# ------------------------------------------------------------
# GROUND TRUTH
# ------------------------------------------------------------

GROUND_TRUTH_PATH = (
    "/content/drive/MyDrive/stage3_normalized/"
    "Copy of train_ground_truth.tsv"
)


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_DIR = (
    "/content/stage4_blocking"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# DUCKDB DATABASE
# ------------------------------------------------------------

DB_PATH = os.path.join(
    OUTPUT_DIR,
    "stage4_blocking.duckdb"
)


# ------------------------------------------------------------
# PERFORMANCE
# ------------------------------------------------------------

DUCKDB_THREADS = 4


# ------------------------------------------------------------
# BLOCKING
# ------------------------------------------------------------

MAX_BLOCK_SIZE = 1000


# ------------------------------------------------------------
# TF-IDF
# ------------------------------------------------------------

TFIDF_TOP_K = 15

TFIDF_MAX_FEATURES = 50_000

TFIDF_BUCKET_MAX_ROWS = 120_000

TFIDF_MIN_DF = 1


# ------------------------------------------------------------
# TF-IDF is applied when deterministic candidates are below
# this number.
# ------------------------------------------------------------

MAX_DETERMINISTIC_CANDIDATES = 30


# ============================================================
# 4. INPUT FILE PATHS
# ============================================================

S1_TRAIN = os.path.join(
    TRAIN_STAGE3_DIR,
    "train_source1_stage3_normalized.tsv"
)

S2_TRAIN = os.path.join(
    TRAIN_STAGE3_DIR,
    "train_source2_stage3_normalized.tsv"
)

# CORRECT S3 PATH
S3_TRAIN = os.path.join(
    TRAIN_STAGE3_DIR,
    "train_source3_stage3_normalized.tsv"
)


S1_TEST = os.path.join(
    TEST_STAGE3_DIR,
    "test_source1_stage3_normalized.tsv"
)

S2_TEST = os.path.join(
    TEST_STAGE3_DIR,
    "test_source2_stage3_normalized.tsv"
)

S3_TEST = os.path.join(
    TEST_STAGE3_DIR,
    "Copy of test_source3_stage3_normalized.tsv"
)


GT_TRAIN = GROUND_TRUTH_PATH


# ============================================================
# 5. FILE EXISTENCE CHECK
# ============================================================
#
# ONLY checks whether the required file exists.
#
# NO ROW COUNT CHECK.
# ============================================================

required_files = {
    "train_source1": S1_TRAIN,
    "train_source2": S2_TRAIN,
    "train_source3": S3_TRAIN,
    "test_source1": S1_TEST,
    "test_source2": S2_TEST,
    "test_source3": S3_TEST,
    "train_ground_truth": GT_TRAIN,
}


print("=" * 100)
print("AMAZON ML CHALLENGE 2026 — STAGE 4")
print("=" * 100)

print("\nChecking required files:\n")


missing_files = []

for name, path in required_files.items():

    if os.path.exists(path):

        size_gb = (
            os.path.getsize(path)
            / (1024 ** 3)
        )

        print(
            f"✓ {name:<20} "
            f"{size_gb:>8.2f} GB"
        )

    else:

        print(
            f"✗ {name:<20} NOT FOUND"
        )

        missing_files.append(path)


if missing_files:

    print("\nMissing files:\n")

    for path in missing_files:
        print(" -", path)

    raise FileNotFoundError(
        "\nRequired files are missing."
    )


print("\nAll required files found.")


# ============================================================
# 6. CLOSE PREVIOUS DUCKDB CONNECTION IF PRESENT
# ============================================================

if "con" in globals():

    try:
        con.close()
    except:
        pass


# ============================================================
# 7. RESET PREVIOUS DUCKDB DATABASE
# ============================================================

if os.path.exists(DB_PATH):

    try:
        os.remove(DB_PATH)
    except PermissionError:

        print(
            "DuckDB database is currently locked."
        )

        raise


# ============================================================
# 8. HELPER FUNCTIONS
# ============================================================

def escape_sql_string(value):

    return str(value).replace(
        "'",
        "''"
    )


def sql_ident(name):

    return '"' + str(name).replace(
        '"',
        '""'
    ) + '"'


def get_columns(
    con,
    table_name
):

    rows = con.execute(
        f'DESCRIBE {sql_ident(table_name)}'
    ).fetchall()

    return [
        row[0]
        for row in rows
    ]


def find_column(
    columns,
    candidates,
    required=True,
    label="column"
):

    lookup = {
        str(column).strip().lower(): column
        for column in columns
    }

    for candidate in candidates:

        key = (
            str(candidate)
            .strip()
            .lower()
        )

        if key in lookup:

            return lookup[key]


    if required:

        raise KeyError(
            "\nRequired "
            f"{label} not found.\n\n"
            f"Available columns:\n{columns}\n\n"
            f"Tried:\n{candidates}"
        )


    return None


def clean_sql_value(
    column_name
):

    if column_name is None:

        return "''"


    return (
        f"lower(trim(coalesce("
        f"{sql_ident(column_name)}, "
        f"'')))"
    )


# ============================================================
# 9. CONNECT TO DUCKDB
# ============================================================

print(
    "\nConnecting to DuckDB..."
)

con = duckdb.connect(
    DB_PATH
)

con.execute(
    f"PRAGMA threads={DUCKDB_THREADS}"
)

con.execute(
    "PRAGMA enable_progress_bar=false"
)


# ============================================================
# 10. LOAD TSV INTO DUCKDB
# ============================================================

def load_tsv(
    con,
    table_name,
    path
):

    print(
        f"\nLoading: {table_name}"
    )

    print(
        path
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE
        {sql_ident(table_name)}
        AS

        SELECT *

        FROM read_csv(
            '{escape_sql_string(path)}',

            delim='\\t',

            header=true,

            quote='"',

            escape='"',

            all_varchar=true,

            null_padding=true,

            ignore_errors=false
        );
        """
    )


    row_count = con.execute(
        f"""
        SELECT COUNT(*)

        FROM {sql_ident(table_name)}
        """
    ).fetchone()[0]


    print(
        f"{table_name}: "
        f"{row_count:,} rows"
    )


    return row_count


# ============================================================
# 11. LOAD ALL DATA
# ============================================================

print(
    "\n========== LOADING DATA =========="
)


load_tsv(
    con,
    "train_s1",
    S1_TRAIN
)

load_tsv(
    con,
    "train_s2",
    S2_TRAIN
)

load_tsv(
    con,
    "train_s3",
    S3_TRAIN
)

load_tsv(
    con,
    "test_s1",
    S1_TEST
)

load_tsv(
    con,
    "test_s2",
    S2_TEST
)

load_tsv(
    con,
    "test_s3",
    S3_TEST
)

load_tsv(
    con,
    "train_gt",
    GT_TRAIN
)


# ============================================================
# 12. DETECT ACTUAL COLUMNS
# ============================================================

print(
    "\n========== DETECTING COLUMNS =========="
)


train_s1_cols = get_columns(
    con,
    "train_s1"
)

train_s2_cols = get_columns(
    con,
    "train_s2"
)

train_s3_cols = get_columns(
    con,
    "train_s3"
)

test_s1_cols = get_columns(
    con,
    "test_s1"
)

test_s2_cols = get_columns(
    con,
    "test_s2"
)

test_s3_cols = get_columns(
    con,
    "test_s3"
)


# ============================================================
# 13. ENTITY IDS
# ============================================================

train_s1_id = find_column(
    train_s1_cols,
    [
        "entity_id",
        "source1_entity_id",
    ],
    True,
    "TRAIN S1 entity ID"
)

train_s2_id = find_column(
    train_s2_cols,
    [
        "entity_id",
        "source2_entity_id",
    ],
    True,
    "TRAIN S2 entity ID"
)

train_s3_id = find_column(
    train_s3_cols,
    [
        "entity_id",
        "source3_entity_id",
    ],
    True,
    "TRAIN S3 entity ID"
)


test_s1_id = find_column(
    test_s1_cols,
    [
        "entity_id",
        "source1_entity_id",
    ],
    True,
    "TEST S1 entity ID"
)

test_s2_id = find_column(
    test_s2_cols,
    [
        "entity_id",
        "source2_entity_id",
    ],
    True,
    "TEST S2 entity ID"
)

test_s3_id = find_column(
    test_s3_cols,
    [
        "entity_id",
        "source3_entity_id",
    ],
    True,
    "TEST S3 entity ID"
)


# ============================================================
# 14. BUSINESS NAME
# ============================================================
#
# ACTUAL STAGE-3 COLUMN:
#
# business_name_normalized_original
#
# Created by Stage 3 from business_name_basic.
# ============================================================

def find_name_column(
    columns,
    label
):

    return find_column(
        columns,

        [
            "business_name_normalized_original",

            # Compatibility names
            "business_name_normalized",
            "business_name_norm",
            "business_name_clean_normalized",

            # Older / fallback names
            "business_name_basic",
            "business_name_original",
            "business_name",
        ],

        True,

        f"{label} business name"
    )


train_s1_name = find_name_column(
    train_s1_cols,
    "TRAIN S1"
)

train_s2_name = find_name_column(
    train_s2_cols,
    "TRAIN S2"
)

train_s3_name = find_name_column(
    train_s3_cols,
    "TRAIN S3"
)

test_s1_name = find_name_column(
    test_s1_cols,
    "TEST S1"
)

test_s2_name = find_name_column(
    test_s2_cols,
    "TEST S2"
)

test_s3_name = find_name_column(
    test_s3_cols,
    "TEST S3"
)


# ============================================================
# 15. ROMAN NAME
# ============================================================
#
# ACTUAL STAGE-3 COLUMN:
#
# business_name_normalized_roman
# ============================================================

def find_roman_column(
    columns,
    label
):

    return find_column(
        columns,

        [
            "business_name_normalized_roman",

            "business_name_roman_search",
            "business_name_roman",
            "business_name_romanized",
        ],

        True,

        f"{label} Roman name"
    )


train_s1_roman = find_roman_column(
    train_s1_cols,
    "TRAIN S1"
)

train_s2_roman = find_roman_column(
    train_s2_cols,
    "TRAIN S2"
)

train_s3_roman = find_roman_column(
    train_s3_cols,
    "TRAIN S3"
)

test_s1_roman = find_roman_column(
    test_s1_cols,
    "TEST S1"
)

test_s2_roman = find_roman_column(
    test_s2_cols,
    "TEST S2"
)

test_s3_roman = find_roman_column(
    test_s3_cols,
    "TEST S3"
)


# ============================================================
# 16. PHONETIC NAME
# ============================================================
#
# ACTUAL STAGE-3 COLUMN:
#
# business_name_roman_phonetic
# ============================================================

def find_phonetic_column(
    columns,
    label
):

    return find_column(
        columns,

        [
            "business_name_roman_phonetic",
            "business_name_phonetic",
        ],

        False,

        f"{label} phonetic name"
    )


train_s1_phonetic = find_phonetic_column(
    train_s1_cols,
    "TRAIN S1"
)

train_s2_phonetic = find_phonetic_column(
    train_s2_cols,
    "TRAIN S2"
)

train_s3_phonetic = find_phonetic_column(
    train_s3_cols,
    "TRAIN S3"
)

test_s1_phonetic = find_phonetic_column(
    test_s1_cols,
    "TEST S1"
)

test_s2_phonetic = find_phonetic_column(
    test_s2_cols,
    "TEST S2"
)

test_s3_phonetic = find_phonetic_column(
    test_s3_cols,
    "TEST S3"
)


# ============================================================
# 17. COUNTRY
# ============================================================

def find_country_column(
    columns,
    label
):

    return find_column(
        columns,

        [
            "country",
            "country_normalized",
        ],

        True,

        f"{label} country"
    )


train_s1_country = find_country_column(
    train_s1_cols,
    "TRAIN S1"
)

train_s2_country = find_country_column(
    train_s2_cols,
    "TRAIN S2"
)

train_s3_country = find_country_column(
    train_s3_cols,
    "TRAIN S3"
)

test_s1_country = find_country_column(
    test_s1_cols,
    "TEST S1"
)

test_s2_country = find_country_column(
    test_s2_cols,
    "TEST S2"
)

test_s3_country = find_country_column(
    test_s3_cols,
    "TEST S3"
)


# ============================================================
# 18. ADDRESS COLUMNS
# ============================================================
#
# ACTUAL STAGE-3 COLUMNS:
#
# address_house_number
# address_postal_code
# address_locality
# address_city
# address_state_region
# ============================================================

def find_optional_column(
    columns,
    candidates
):

    return find_column(
        columns,
        candidates,
        False
    )


def detect_address_columns(
    columns,
    label
):

    result = {

        "postal": find_optional_column(
            columns,
            [
                "address_postal_code",
                "business_address_postal_code",
                "postal_code",
            ]
        ),

        "house": find_optional_column(
            columns,
            [
                "address_house_number",
                "business_address_house_number",
                "house_number",
            ]
        ),

        "locality": find_optional_column(
            columns,
            [
                "address_locality",
                "business_address_locality",
                "locality",
            ]
        ),

        "city": find_optional_column(
            columns,
            [
                "address_city",
                "business_address_city",
                "city",
            ]
        ),

        "state": find_optional_column(
            columns,
            [
                "address_state_region",
                "business_address_state_region",
                "state_region",
                "state",
                "province",
            ]
        ),
    }


    return result


train_s1_address = detect_address_columns(
    train_s1_cols,
    "TRAIN S1"
)

train_s2_address = detect_address_columns(
    train_s2_cols,
    "TRAIN S2"
)

train_s3_address = detect_address_columns(
    train_s3_cols,
    "TRAIN S3"
)

test_s1_address = detect_address_columns(
    test_s1_cols,
    "TEST S1"
)

test_s2_address = detect_address_columns(
    test_s2_cols,
    "TEST S2"
)

test_s3_address = detect_address_columns(
    test_s3_cols,
    "TEST S3"
)


# ============================================================
# 19. DISPLAY SELECTED COLUMN MAPPING
# ============================================================

print(
    "\n========== FINAL COLUMN MAPPING =========="
)


print("\nTRAIN S1")
print("ID       :", train_s1_id)
print("NAME     :", train_s1_name)
print("ROMAN    :", train_s1_roman)
print("PHONETIC :", train_s1_phonetic)
print("COUNTRY  :", train_s1_country)
print("ADDRESS  :", train_s1_address)


print("\nTRAIN S2")
print("ID       :", train_s2_id)
print("NAME     :", train_s2_name)
print("ROMAN    :", train_s2_roman)
print("PHONETIC :", train_s2_phonetic)
print("COUNTRY  :", train_s2_country)
print("ADDRESS  :", train_s2_address)


print("\nTRAIN S3")
print("ID       :", train_s3_id)
print("NAME     :", train_s3_name)
print("ROMAN    :", train_s3_roman)
print("PHONETIC :", train_s3_phonetic)
print("COUNTRY  :", train_s3_country)
print("ADDRESS  :", train_s3_address)


print("\nTEST S1")
print("ID       :", test_s1_id)
print("NAME     :", test_s1_name)
print("ROMAN    :", test_s1_roman)
print("PHONETIC :", test_s1_phonetic)
print("COUNTRY  :", test_s1_country)
print("ADDRESS  :", test_s1_address)


print("\nTEST S2")
print("ID       :", test_s2_id)
print("NAME     :", test_s2_name)
print("ROMAN    :", test_s2_roman)
print("PHONETIC :", test_s2_phonetic)
print("COUNTRY  :", test_s2_country)
print("ADDRESS  :", test_s2_address)


print("\nTEST S3")
print("ID       :", test_s3_id)
print("NAME     :", test_s3_name)
print("ROMAN    :", test_s3_roman)
print("PHONETIC :", test_s3_phonetic)
print("COUNTRY  :", test_s3_country)
print("ADDRESS  :", test_s3_address)


# ============================================================
# 20. CREATE COMPACT FEATURE TABLE
# ============================================================

def create_feature_table(
    con,
    source_table,
    output_table,
    id_col,
    name_col,
    roman_col,
    phonetic_col,
    country_col,
    address_info
):

    name_expr = clean_sql_value(
        name_col
    )

    roman_expr = clean_sql_value(
        roman_col
    )

    phonetic_expr = clean_sql_value(
        phonetic_col
    )

    country_expr = clean_sql_value(
        country_col
    )

    postal_expr = clean_sql_value(
        address_info.get("postal")
    )

    house_expr = clean_sql_value(
        address_info.get("house")
    )

    locality_expr = clean_sql_value(
        address_info.get("locality")
    )

    city_expr = clean_sql_value(
        address_info.get("city")
    )

    state_expr = clean_sql_value(
        address_info.get("state")
    )


    sql = f"""
    CREATE OR REPLACE TABLE
    {sql_ident(output_table)}
    AS

    SELECT

        CAST(
            {sql_ident(id_col)}
            AS VARCHAR
        ) AS entity_id,


        {name_expr}
        AS name_key_base,


        {roman_expr}
        AS roman_name_key_base,


        {phonetic_expr}
        AS phonetic_key_base,


        {country_expr}
        AS country_key,


        {postal_expr}
        AS postal_key,


        {house_expr}
        AS house_key,


        {locality_expr}
        AS locality_key,


        {city_expr}
        AS city_key,


        {state_expr}
        AS state_key,


        concat_ws(
            '¦',

            nullif(
                {house_expr},
                ''
            ),

            nullif(
                {locality_expr},
                ''
            )
        ) AS house_locality_key,


        concat_ws(
            '¦',

            nullif(
                {house_expr},
                ''
            ),

            nullif(
                {city_expr},
                ''
            )
        ) AS house_city_key,


        concat_ws(
            '¦',

            nullif(
                {name_expr},
                ''
            ),

            nullif(
                {country_expr},
                ''
            )
        ) AS name_country_key,


        concat_ws(
            '¦',

            nullif(
                {name_expr},
                ''
            ),

            nullif(
                {postal_expr},
                ''
            )
        ) AS name_postal_key,


        regexp_extract(
            {roman_expr},
            '([a-z0-9]{{2}})',
            1
        ) AS tfidf_prefix2


    FROM {sql_ident(source_table)};
    """


    con.execute(
        sql
    )


# ============================================================
# 21. CREATE TRAIN FEATURE TABLES
# ============================================================

print(
    "\n========== CREATING TRAIN FEATURE TABLES =========="
)


create_feature_table(
    con,
    "train_s1",
    "f_train_s1",
    train_s1_id,
    train_s1_name,
    train_s1_roman,
    train_s1_phonetic,
    train_s1_country,
    train_s1_address
)


create_feature_table(
    con,
    "train_s2",
    "f_train_s2",
    train_s2_id,
    train_s2_name,
    train_s2_roman,
    train_s2_phonetic,
    train_s2_country,
    train_s2_address
)


create_feature_table(
    con,
    "train_s3",
    "f_train_s3",
    train_s3_id,
    train_s3_name,
    train_s3_roman,
    train_s3_phonetic,
    train_s3_country,
    train_s3_address
)


# ============================================================
# 22. CREATE TEST FEATURE TABLES
# ============================================================

print(
    "\n========== CREATING TEST FEATURE TABLES =========="
)


create_feature_table(
    con,
    "test_s1",
    "f_test_s1",
    test_s1_id,
    test_s1_name,
    test_s1_roman,
    test_s1_phonetic,
    test_s1_country,
    test_s1_address
)


create_feature_table(
    con,
    "test_s2",
    "f_test_s2",
    test_s2_id,
    test_s2_name,
    test_s2_roman,
    test_s2_phonetic,
    test_s2_country,
    test_s2_address
)


create_feature_table(
    con,
    "test_s3",
    "f_test_s3",
    test_s3_id,
    test_s3_name,
    test_s3_roman,
    test_s3_phonetic,
    test_s3_country,
    test_s3_address
)


# ============================================================
# 23. FEATURE TABLE COUNTS
# ============================================================

print(
    "\n========== FEATURE TABLES =========="
)


for table in [
    "f_train_s1",
    "f_train_s2",
    "f_train_s3",
    "f_test_s1",
    "f_test_s2",
    "f_test_s3",
]:

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {table}
        """
    ).fetchone()[0]

    print(
        f"{table:<15}: "
        f"{count:,} rows"
    )


# ============================================================
# 24. CREATE CANDIDATE TABLES
# ============================================================

con.execute(
    """
    CREATE OR REPLACE TABLE
    train_candidate_long (

        source1_entity_id VARCHAR,

        candidate_entity_id VARCHAR,

        source_pair VARCHAR,

        strategy VARCHAR
    )
    """
)


con.execute(
    """
    CREATE OR REPLACE TABLE
    test_candidate_long (

        source1_entity_id VARCHAR,

        candidate_entity_id VARCHAR,

        source_pair VARCHAR,

        strategy VARCHAR
    )
    """
)


# ============================================================
# 25. EXACT BLOCKING
# ============================================================

def add_exact_block(
    s1_table,
    sx_table,
    target_table,
    source_pair,
    key_column,
    strategy
):

    print(
        f"Block: "
        f"{strategy} "
        f"[{source_pair}]"
    )


    con.execute(
        f"""
        INSERT INTO {target_table}

        SELECT

            a.entity_id
                AS source1_entity_id,

            b.entity_id
                AS candidate_entity_id,

            '{source_pair}'
                AS source_pair,

            '{strategy}'
                AS strategy


        FROM {s1_table} a


        INNER JOIN {sx_table} b

            ON
                a.{sql_ident(key_column)} <> ''

                AND

                a.{sql_ident(key_column)}
                =
                b.{sql_ident(key_column)}


        WHERE

            a.{sql_ident(key_column)}

            IN
            (

                SELECT
                    {sql_ident(key_column)}

                FROM {s1_table}

                WHERE
                    {sql_ident(key_column)} <> ''

                GROUP BY
                    {sql_ident(key_column)}

                HAVING
                    COUNT(*) <= {MAX_BLOCK_SIZE}
            )


            AND


            b.{sql_ident(key_column)}

            IN
            (

                SELECT
                    {sql_ident(key_column)}

                FROM {sx_table}

                WHERE
                    {sql_ident(key_column)} <> ''

                GROUP BY
                    {sql_ident(key_column)}

                HAVING
                    COUNT(*) <= {MAX_BLOCK_SIZE}
            );
        """
    )


# ============================================================
# 26. HOUSE + LOCALITY / CITY BLOCK
# ============================================================

def add_house_location_block(
    s1_table,
    sx_table,
    target_table,
    source_pair
):

    print(
        f"Block: "
        f"house_location "
        f"[{source_pair}]"
    )


    con.execute(
        f"""
        INSERT INTO {target_table}

        SELECT

            a.entity_id
                AS source1_entity_id,

            b.entity_id
                AS candidate_entity_id,

            '{source_pair}'
                AS source_pair,

            'house_locality'
                AS strategy


        FROM {s1_table} a


        INNER JOIN {sx_table} b

            ON

            (

                a.house_locality_key <> ''

                AND

                a.house_locality_key
                =
                b.house_locality_key

            )

            OR

            (

                a.house_city_key <> ''

                AND

                a.house_city_key
                =
                b.house_city_key

            )


        WHERE

        (

            a.house_locality_key <> ''

            AND

            a.house_locality_key IN
            (

                SELECT
                    house_locality_key

                FROM {s1_table}

                WHERE
                    house_locality_key <> ''

                GROUP BY
                    house_locality_key

                HAVING
                    COUNT(*) <= {MAX_BLOCK_SIZE}
            )

            AND

            b.house_locality_key IN
            (

                SELECT
                    house_locality_key

                FROM {sx_table}

                WHERE
                    house_locality_key <> ''

                GROUP BY
                    house_locality_key

                HAVING
                    COUNT(*) <= {MAX_BLOCK_SIZE}
            )

        )

        OR

        (

            a.house_city_key <> ''

            AND

            a.house_city_key IN
            (

                SELECT
                    house_city_key

                FROM {s1_table}

                WHERE
                    house_city_key <> ''

                GROUP BY
                    house_city_key

                HAVING
                    COUNT(*) <= {MAX_BLOCK_SIZE}
            )

            AND

            b.house_city_key IN
            (

                SELECT
                    house_city_key

                FROM {sx_table}

                WHERE
                    house_city_key <> ''

                GROUP BY
                    house_city_key

                HAVING
                    COUNT(*) <= {MAX_BLOCK_SIZE}
            )

        );
        """
    )


# ============================================================
# 27. TRAIN DETERMINISTIC BLOCKING
# ============================================================

print(
    "\n========== TRAIN DETERMINISTIC BLOCKING =========="
)


train_pairs = [

    (
        "f_train_s2",
        "s1_s2"
    ),

    (
        "f_train_s3",
        "s1_s3"
    ),

]


# ------------------------------------------------------------
# Strategy 1 — Exact normalized name
# ------------------------------------------------------------

for sx_table, pair in train_pairs:

    add_exact_block(
        "f_train_s1",
        sx_table,
        "train_candidate_long",
        pair,
        "name_key_base",
        "exact_name"
    )


# ------------------------------------------------------------
# Strategy 2 — Name + country
# ------------------------------------------------------------

for sx_table, pair in train_pairs:

    add_exact_block(
        "f_train_s1",
        sx_table,
        "train_candidate_long",
        pair,
        "name_country_key",
        "name_country"
    )


# ------------------------------------------------------------
# Strategy 3 — Name + postal
# ------------------------------------------------------------

for sx_table, pair in train_pairs:

    add_exact_block(
        "f_train_s1",
        sx_table,
        "train_candidate_long",
        pair,
        "name_postal_key",
        "name_postal"
    )


# ------------------------------------------------------------
# Strategy 4 — House + locality / city
# ------------------------------------------------------------

for sx_table, pair in train_pairs:

    add_house_location_block(
        "f_train_s1",
        sx_table,
        "train_candidate_long",
        pair
    )


# ------------------------------------------------------------
# Strategy 5 — Phonetic name
# ------------------------------------------------------------

for sx_table, pair in train_pairs:

    if (
        train_s1_phonetic is not None
        and
        (
            (
                pair == "s1_s2"
                and train_s2_phonetic is not None
            )
            or
            (
                pair == "s1_s3"
                and train_s3_phonetic is not None
            )
        )
    ):

        add_exact_block(
            "f_train_s1",
            sx_table,
            "train_candidate_long",
            pair,
            "phonetic_key_base",
            "phonetic_name"
        )


# ------------------------------------------------------------
# Strategy 6 — Exact Roman/transliterated name
# ------------------------------------------------------------

for sx_table, pair in train_pairs:

    add_exact_block(
        "f_train_s1",
        sx_table,
        "train_candidate_long",
        pair,
        "roman_name_key_base",
        "roman_exact_name"
    )


# ============================================================
# 28. TEST DETERMINISTIC BLOCKING
# ============================================================

print(
    "\n========== TEST DETERMINISTIC BLOCKING =========="
)


test_pairs = [

    (
        "f_test_s2",
        "s1_s2"
    ),

    (
        "f_test_s3",
        "s1_s3"
    ),

]


# ------------------------------------------------------------
# Strategy 1 — Exact normalized name
# ------------------------------------------------------------

for sx_table, pair in test_pairs:

    add_exact_block(
        "f_test_s1",
        sx_table,
        "test_candidate_long",
        pair,
        "name_key_base",
        "exact_name"
    )


# ------------------------------------------------------------
# Strategy 2 — Name + country
# ------------------------------------------------------------

for sx_table, pair in test_pairs:

    add_exact_block(
        "f_test_s1",
        sx_table,
        "test_candidate_long",
        pair,
        "name_country_key",
        "name_country"
    )


# ------------------------------------------------------------
# Strategy 3 — Name + postal
# ------------------------------------------------------------

for sx_table, pair in test_pairs:

    add_exact_block(
        "f_test_s1",
        sx_table,
        "test_candidate_long",
        pair,
        "name_postal_key",
        "name_postal"
    )


# ------------------------------------------------------------
# Strategy 4 — House + locality / city
# ------------------------------------------------------------

for sx_table, pair in test_pairs:

    add_house_location_block(
        "f_test_s1",
        sx_table,
        "test_candidate_long",
        pair
    )


# ------------------------------------------------------------
# Strategy 5 — Phonetic name
# ------------------------------------------------------------

for sx_table, pair in test_pairs:

    if (
        test_s1_phonetic is not None
        and
        (
            (
                pair == "s1_s2"
                and test_s2_phonetic is not None
            )
            or
            (
                pair == "s1_s3"
                and test_s3_phonetic is not None
            )
        )
    ):

        add_exact_block(
            "f_test_s1",
            sx_table,
            "test_candidate_long",
            pair,
            "phonetic_key_base",
            "phonetic_name"
        )


# ------------------------------------------------------------
# Strategy 6 — Exact Roman/transliterated name
# ------------------------------------------------------------

for sx_table, pair in test_pairs:

    add_exact_block(
        "f_test_s1",
        sx_table,
        "test_candidate_long",
        pair,
        "roman_name_key_base",
        "roman_exact_name"
    )


# ============================================================
# 29. GET UNRESOLVED S1 RECORDS
# ============================================================

def get_unresolved_s1(
    s1_feature_table,
    candidate_table
):

    return con.execute(
        f"""
        SELECT

            a.entity_id,

            a.roman_name_key_base,

            a.tfidf_prefix2,

            a.country_key

        FROM {s1_feature_table} a


        LEFT JOIN
        (

            SELECT

                source1_entity_id,

                COUNT(
                    DISTINCT candidate_entity_id
                ) AS candidate_count

            FROM {candidate_table}

            GROUP BY
                source1_entity_id

        ) c


        ON

            a.entity_id
            =
            c.source1_entity_id


        WHERE

            COALESCE(
                c.candidate_count,
                0
            )
            <
            {MAX_DETERMINISTIC_CANDIDATES}


            AND

            a.roman_name_key_base <> '';
        """
    ).df()


# ============================================================
# 30. TF-IDF BLOCKING
# ============================================================

def run_tfidf_block(
    s1_feature_table,
    sx_feature_table,
    candidate_table,
    source_pair
):

    print(
        f"\n========== TF-IDF: "
        f"{source_pair} =========="
    )


    unresolved = get_unresolved_s1(
        s1_feature_table,
        candidate_table
    )


    print(
        "S1 records requiring TF-IDF:",
        f"{len(unresolved):,}"
    )


    if unresolved.empty:

        return


    # --------------------------------------------------------
    # Country + first 2 Roman characters
    # --------------------------------------------------------

    unresolved["bucket"] = (

        unresolved["country_key"]
        .fillna("")
        .astype(str)

        + "¦" +

        unresolved["tfidf_prefix2"]
        .fillna("")
        .astype(str)

    )


    buckets = (
        unresolved["bucket"]
        .value_counts()
        .index
        .tolist()
    )


    inserted = 0

    skipped_large = 0

    start_time = time.time()


    # ========================================================
    # Process buckets
    # ========================================================

    for bucket_number, bucket in enumerate(
        buckets,
        start=1
    ):

        s1_bucket = unresolved[
            unresolved["bucket"]
            ==
            bucket
        ]


        if s1_bucket.empty:

            continue


        country, prefix = bucket.split(
            "¦",
            1
        )


        # ----------------------------------------------------
        # Source rows in same country + prefix
        # ----------------------------------------------------

        source_df = con.execute(
            f"""
            SELECT

                entity_id,

                roman_name_key_base

            FROM {sx_feature_table}

            WHERE

                roman_name_key_base <> ''

                AND

                country_key = ?

                AND

                tfidf_prefix2 = ?
            """,
            [
                country,
                prefix
            ]
        ).df()


        if source_df.empty:

            continue


        # ----------------------------------------------------
        # Bucket size limit
        # ----------------------------------------------------

        if (
            len(source_df)
            >
            TFIDF_BUCKET_MAX_ROWS
        ):

            skipped_large += 1

            continue


        corpus = (
            source_df[
                "roman_name_key_base"
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )


        queries = (
            s1_bucket[
                "roman_name_key_base"
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )


        try:

            # ------------------------------------------------
            # Character TF-IDF
            # ------------------------------------------------

            vectorizer = TfidfVectorizer(

                analyzer="char",

                ngram_range=(
                    3,
                    5
                ),

                min_df=TFIDF_MIN_DF,

                max_features=TFIDF_MAX_FEATURES,

                lowercase=False,

                dtype=np.float32
            )


            X = vectorizer.fit_transform(
                corpus
            )


            Q = vectorizer.transform(
                queries
            )


            if X.shape[1] == 0:

                continue


            k = min(
                TFIDF_TOP_K,
                X.shape[0]
            )


            nn = NearestNeighbors(

                metric="cosine",

                algorithm="brute",

                n_neighbors=k,

                n_jobs=-1
            )


            nn.fit(
                X
            )


            distances, indices = nn.kneighbors(
                Q,
                return_distance=True
            )


            candidate_rows = []


            # ------------------------------------------------
            # Convert retrieval results into candidate rows
            # ------------------------------------------------

            for query_index in range(
                len(s1_bucket)
            ):

                source1_id = str(
                    s1_bucket.iloc[
                        query_index
                    ]["entity_id"]
                )


                for neighbor_index in range(
                    indices.shape[1]
                ):

                    source_index = int(
                        indices[
                            query_index,
                            neighbor_index
                        ]
                    )


                    candidate_id = str(
                        source_df.iloc[
                            source_index
                        ]["entity_id"]
                    )


                    similarity = (
                        1.0
                        -
                        float(
                            distances[
                                query_index,
                                neighbor_index
                            ]
                        )
                    )


                    if similarity <= 0:

                        continue


                    candidate_rows.append(
                        (
                            source1_id,
                            candidate_id,
                            source_pair,
                            "tfidf_char_topk"
                        )
                    )


            # ------------------------------------------------
            # Insert candidates
            # ------------------------------------------------

            if candidate_rows:

                temp_name = (
                    "tmp_tfidf_"
                    f"{source_pair}_"
                    f"{bucket_number}"
                )


                temp_df = pd.DataFrame(

                    candidate_rows,

                    columns=[
                        "source1_entity_id",
                        "candidate_entity_id",
                        "source_pair",
                        "strategy"
                    ]
                )


                con.register(
                    temp_name,
                    temp_df
                )


                con.execute(
                    f"""
                    INSERT INTO
                    {candidate_table}

                    SELECT *

                    FROM {temp_name}
                    """
                )


                con.unregister(
                    temp_name
                )


                inserted += len(
                    candidate_rows
                )


        except Exception as error:

            print(
                f"\nTF-IDF bucket error: "
                f"{bucket}"
            )

            print(error)


        del source_df

        gc.collect()


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            bucket_number % 100 == 0
            or
            bucket_number == len(buckets)
        ):

            elapsed_minutes = (
                time.time()
                -
                start_time
            ) / 60


            print(

                f"Bucket "
                f"{bucket_number:,}/"
                f"{len(buckets):,} | "

                f"Inserted: "
                f"{inserted:,} | "

                f"Large buckets: "
                f"{skipped_large:,} | "

                f"Time: "
                f"{elapsed_minutes:.1f} min"
            )


    print(
        "\nTF-IDF completed."
    )

    print(
        "Inserted candidates:",
        f"{inserted:,}"
    )


    print(
        "Oversized buckets:",
        f"{skipped_large:,}"
    )


# ============================================================
# 31. RUN TF-IDF
# ============================================================

run_tfidf_block(
    "f_train_s1",
    "f_train_s2",
    "train_candidate_long",
    "s1_s2"
)

run_tfidf_block(
    "f_train_s1",
    "f_train_s3",
    "train_candidate_long",
    "s1_s3"
)

run_tfidf_block(
    "f_test_s1",
    "f_test_s2",
    "test_candidate_long",
    "s1_s2"
)

run_tfidf_block(
    "f_test_s1",
    "f_test_s3",
    "test_candidate_long",
    "s1_s3"
)


# ============================================================
# 32. FINAL DEDUPLICATION
# ============================================================

print(
    "\n========== FINAL DEDUPLICATION =========="
)


for table in [
    "train_candidate_long",
    "test_candidate_long"
]:

    con.execute(
        f"""
        CREATE OR REPLACE TABLE
        {table}_final
        AS

        SELECT DISTINCT

            source1_entity_id,

            candidate_entity_id,

            source_pair,

            strategy

        FROM {table};
        """
    )


    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {table}_final
        """
    ).fetchone()[0]


    print(
        f"{table}_final: "
        f"{count:,} rows"
    )


# ============================================================
# 33. STRATEGY AUDIT
# ============================================================

print(
    "\n========== STRATEGY AUDIT =========="
)


for dataset_name in [
    "train",
    "test"
]:

    base_table = (
        f"{dataset_name}_candidate_long_final"
    )


    audit_table = (
        f"{dataset_name}_candidate_strategy_audit"
    )


    con.execute(
        f"""
        CREATE OR REPLACE TABLE
        {audit_table}
        AS

        SELECT

            source_pair,

            strategy,

            COUNT(*) AS candidate_rows,

            COUNT(
                DISTINCT source1_entity_id
            )
            AS source1_with_candidates,

            COUNT(
                DISTINCT candidate_entity_id
            )
            AS unique_candidates

        FROM {base_table}

        GROUP BY

            source_pair,

            strategy

        ORDER BY

            source_pair,

            strategy;
        """
    )


    audit_path = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_candidate_strategy_audit.tsv"
    )


    con.execute(
        f"""
        COPY
        {audit_table}

        TO
        '{escape_sql_string(audit_path)}'

        (
            HEADER,
            DELIMITER '\\t'
        )
        """
    )


    print(
        "Saved:",
        audit_path
    )


# ============================================================
# 34. CANDIDATE COUNT DISTRIBUTION
# ============================================================

print(
    "\n========== CANDIDATE COUNT DISTRIBUTION =========="
)


for dataset_name in [
    "train",
    "test"
]:

    candidate_table = (
        f"{dataset_name}_candidate_long_final"
    )


    s1_table = (
        f"f_{dataset_name}_s1"
    )


    distribution = con.execute(
        f"""
        SELECT

            a.entity_id
                AS source1_entity_id,

            COALESCE(

                COUNT(
                    DISTINCT c.candidate_entity_id
                ),

                0

            ) AS candidate_count


        FROM {s1_table} a


        LEFT JOIN {candidate_table} c

            ON

                a.entity_id
                =
                c.source1_entity_id


        GROUP BY

            a.entity_id;
        """
    ).df()


    print(
        f"\n{dataset_name.upper()} candidate statistics:"
    )


    if not distribution.empty:

        print(
            distribution[
                "candidate_count"
            ]
            .describe(
                percentiles=[
                    0.50,
                    0.90,
                    0.95,
                    0.99
                ]
            )
            .to_string()
        )


    distribution_path = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_candidate_count_distribution.tsv"
    )


    distribution.to_csv(
        distribution_path,
        sep="\t",
        index=False
    )


    print(
        "Saved:",
        distribution_path
    )


    del distribution

    gc.collect()


# ============================================================
# 35. TRAINING GROUND-TRUTH PAIRS
# ============================================================

print(
    "\n========== TRAINING BLOCKING RECALL =========="
)


con.execute(
    """
    CREATE OR REPLACE TABLE
    train_gt_pairs
    AS

    SELECT

        CAST(
            source1_entity_id
            AS VARCHAR
        )
        AS source1_entity_id,


        TRIM(x)
        AS matched_entity_id


    FROM train_gt,


    UNNEST(
        string_split(

            COALESCE(
                matched_entity_ids,
                ''
            ),

            ','
        )
    ) AS t(x)


    WHERE
        TRIM(x) <> '';
    """
)


# ============================================================
# 36. IDENTIFY S2 / S3 GROUND TRUTH
# ============================================================

con.execute(
    """
    CREATE OR REPLACE TABLE
    train_gt_pairs_typed
    AS

    SELECT

        g.source1_entity_id,

        g.matched_entity_id
            AS candidate_entity_id,


        CASE

            WHEN s2.entity_id IS NOT NULL

                THEN 's1_s2'


            WHEN s3.entity_id IS NOT NULL

                THEN 's1_s3'


            ELSE 'unknown'

        END
        AS source_pair


    FROM train_gt_pairs g


    LEFT JOIN f_train_s2 s2

        ON
            g.matched_entity_id
            =
            s2.entity_id


    LEFT JOIN f_train_s3 s3

        ON
            g.matched_entity_id
            =
            s3.entity_id;
    """
)


unknown_gt = con.execute(
    """
    SELECT COUNT(*)

    FROM train_gt_pairs_typed

    WHERE source_pair = 'unknown';
    """
).fetchone()[0]


print(
    "Ground-truth IDs not found in S2/S3:",
    f"{unknown_gt:,}"
)


# ============================================================
# 37. PER-STRATEGY RECALL
# ============================================================

strategy_list = con.execute(
    """
    SELECT DISTINCT
        strategy

    FROM train_candidate_long_final

    ORDER BY strategy;
    """
).fetchall()


recall_records = []


for strategy_row in strategy_list:

    strategy = strategy_row[0]


    rows = con.execute(
        """
        SELECT

            g.source_pair,

            COUNT(*)
                AS gt_pairs,

            COUNT(
                c.candidate_entity_id
            )
                AS recovered_pairs,


            CASE

                WHEN COUNT(*) = 0

                    THEN 0.0


                ELSE

                    COUNT(
                        c.candidate_entity_id
                    )::DOUBLE

                    /

                    COUNT(*)

            END
                AS recall


        FROM train_gt_pairs_typed g


        LEFT JOIN
        train_candidate_long_final c


            ON

                c.source1_entity_id
                =
                g.source1_entity_id


                AND


                c.candidate_entity_id
                =
                g.candidate_entity_id


                AND


                c.source_pair
                =
                g.source_pair


                AND


                c.strategy
                =


        ?


        WHERE

            g.source_pair <> 'unknown'


        GROUP BY

            g.source_pair


        ORDER BY

            g.source_pair;

        """,
        [strategy]
    ).fetchall()


    for row in rows:

        recall_records.append(

            {

                "strategy":
                    strategy,

                "source_pair":
                    row[0],

                "gt_pairs":
                    row[1],

                "recovered_pairs":
                    row[2],

                "recall":
                    row[3]
            }
        )


# ============================================================
# 38. UNION RECALL
# ============================================================

union_rows = con.execute(
    """
    SELECT

        g.source_pair,

        COUNT(*)
            AS gt_pairs,

        COUNT(
            c.candidate_entity_id
        )
            AS recovered_pairs,


        CASE

            WHEN COUNT(*) = 0

                THEN 0.0


            ELSE

                COUNT(
                    c.candidate_entity_id
                )::DOUBLE

                /

                COUNT(*)

        END
            AS recall


    FROM train_gt_pairs_typed g


    LEFT JOIN
    (

        SELECT DISTINCT

            source1_entity_id,

            candidate_entity_id,

            source_pair

        FROM train_candidate_long_final

    ) c


        ON

            c.source1_entity_id
            =
            g.source1_entity_id


            AND


            c.candidate_entity_id
            =
            g.candidate_entity_id


            AND


            c.source_pair
            =
            g.source_pair


    WHERE

        g.source_pair <> 'unknown'


    GROUP BY

        g.source_pair


    ORDER BY

        g.source_pair;
    """
).fetchall()


for row in union_rows:

    recall_records.append(

        {

            "strategy":
                "ALL_UNION",

            "source_pair":
                row[0],

            "gt_pairs":
                row[1],

            "recovered_pairs":
                row[2],

            "recall":
                row[3]
        }
    )


recall_df = pd.DataFrame(
    recall_records
)


recall_path = os.path.join(
    OUTPUT_DIR,
    "train_blocking_recall_report.tsv"
)


recall_df.to_csv(
    recall_path,
    sep="\t",
    index=False
)


print(
    recall_df.to_string(
        index=False
    )
)


print(
    "\nSaved:",
    recall_path
)


# ============================================================
# 39. CREATE ONE-ROW-PER-S1 CANDIDATE FILE
# ============================================================

print(
    "\n========== CANDIDATE FILES =========="
)


for dataset_name in [
    "train",
    "test"
]:

    candidate_table = (
        f"{dataset_name}_candidate_long_final"
    )


    s1_table = (
        f"f_{dataset_name}_s1"
    )


    output_df = con.execute(
        f"""
        SELECT

            a.entity_id
                AS source1_entity_id,


            COALESCE(

                string_agg(

                    DISTINCT
                    c.candidate_entity_id,

                    ','

                    ORDER BY
                    c.candidate_entity_id
                ),

                ''

            )
            AS candidate_entity_ids


        FROM {s1_table} a


        LEFT JOIN
        {candidate_table} c


            ON

                a.entity_id
                =
                c.source1_entity_id


        GROUP BY

            a.entity_id


        ORDER BY

            a.entity_id;
        """
    ).df()


    output_path = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_candidate_pairs.tsv"
    )


    output_df.to_csv(
        output_path,
        sep="\t",
        index=False
    )


    candidate_count = (
        output_df[
            "candidate_entity_ids"
        ]
        .fillna("")
        .ne("")
        .sum()
    )


    print(
        "\nSaved:",
        output_path
    )


    print(
        "Rows:",
        f"{len(output_df):,}"
    )


    print(
        "S1 with candidates:",
        f"{candidate_count:,}"
    )


    del output_df

    gc.collect()


# ============================================================
# 40. FINAL TEST candidate_pairs.tsv
# ============================================================

test_candidate_path = os.path.join(
    OUTPUT_DIR,
    "test_candidate_pairs.tsv"
)


candidate_submission_path = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)


print(
    "\nCreating final candidate_pairs.tsv..."
)


with open(
    test_candidate_path,
    "rb"
) as src:

    with open(
        candidate_submission_path,
        "wb"
    ) as dst:

        while True:

            chunk = src.read(
                8 * 1024 * 1024
            )


            if not chunk:

                break


            dst.write(
                chunk
            )


print(
    "Final candidate file:",
    candidate_submission_path
)


# ============================================================
# 41. FINAL SANITY CHECK
# ============================================================

print(
    "\n========== FINAL SANITY CHECK =========="
)


test_s1_rows = con.execute(
    """
    SELECT COUNT(*)

    FROM f_test_s1
    """
).fetchone()[0]


candidate_file_rows = (

    sum(

        1

        for _ in open(
            candidate_submission_path,
            "r",
            encoding="utf-8"
        )

    )

    - 1
)


print(
    "Test S1 rows:",
    f"{test_s1_rows:,}"
)


print(
    "candidate_pairs.tsv rows:",
    f"{candidate_file_rows:,}"
)


# ------------------------------------------------------------
# Duplicate S1 check
# ------------------------------------------------------------

duplicate_s1 = con.execute(
    """
    SELECT COUNT(*)

    FROM
    (

        SELECT

            source1_entity_id

        FROM read_csv(
            ?,

            delim='\\t',

            header=true,

            all_varchar=true
        )


        GROUP BY
            source1_entity_id


        HAVING
            COUNT(*) > 1

    );
    """,
    [candidate_submission_path]
).fetchone()[0]


print(
    "Duplicate S1 rows:",
    f"{duplicate_s1:,}"
)


# ============================================================
# 42. SAVE SUMMARY JSON
# ============================================================

summary = {

    "output_dir":
        OUTPUT_DIR,


    "candidate_submission":
        candidate_submission_path,


    "train_candidate_pairs":
        os.path.join(
            OUTPUT_DIR,
            "train_candidate_pairs.tsv"
        ),


    "test_candidate_pairs":
        os.path.join(
            OUTPUT_DIR,
            "test_candidate_pairs.tsv"
        ),


    "train_strategy_audit":
        os.path.join(
            OUTPUT_DIR,
            "train_candidate_strategy_audit.tsv"
        ),


    "test_strategy_audit":
        os.path.join(
            OUTPUT_DIR,
            "test_candidate_strategy_audit.tsv"
        ),


    "train_blocking_recall_report":
        recall_path,


    "test_s1_rows":
        int(test_s1_rows),


    "candidate_pairs_rows":
        int(candidate_file_rows),


    "duplicate_s1_rows":
        int(duplicate_s1),


    "row_count_completeness_check":
        "DISABLED"
}


summary_path = os.path.join(
    OUTPUT_DIR,
    "stage4_summary.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=2
    )


print(
    "\nSaved:",
    summary_path
)


# ============================================================
# 43. COMPLETION
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "STAGE 4 COMPLETE"
)

print(
    "=" * 100
)


print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)


print(
    "\nFinal candidate file:"
)

print(
    candidate_submission_path
)


print(
    "\nMain files generated:"
)

print(
    "1. train_candidate_pairs.tsv"
)

print(
    "2. test_candidate_pairs.tsv"
)

print(
    "3. candidate_pairs.tsv"
)

print(
    "4. train_candidate_strategy_audit.tsv"
)

print(
    "5. test_candidate_strategy_audit.tsv"
)

print(
    "6. train_blocking_recall_report.tsv"
)

print(
    "7. train_candidate_count_distribution.tsv"
)

print(
    "8. test_candidate_count_distribution.tsv"
)

print(
    "9. stage4_summary.json"
)

print(
    "\nNext stage:"
)

print(
    "Stage 5 — Pair Feature Generation"
)

print(
    "=" * 100
)


# ============================================================
# 44. CLOSE DUCKDB
# ============================================================

con.close()
