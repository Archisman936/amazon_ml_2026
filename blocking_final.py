# ============================================================
# AMAZON ML CHALLENGE 2026
# STAGE 4 — BLOCKING / CANDIDATE GENERATION
# REVISED FINAL VERSION
# ============================================================
#
# IMPORTANT
# ------------------------------------------------------------
# This is a COMPLETE RE-RUN of Stage 4.
#
# It starts from the VERIFIED Stage-3 files.
#
# It does NOT use the old broken Stage-4 files.
#
# OUTPUT:
#
# /content/drive/MyDrive/Colab Notebooks/dataset/stage_4_revised/
#
#     train_candidate_pairs.tsv
#     test_candidate_pairs.tsv
#     candidate_pairs.tsv
#     train_candidate_strategy_audit.tsv
#     test_candidate_strategy_audit.tsv
#     train_candidate_count_distribution.tsv
#     test_candidate_count_distribution.tsv
#     train_blocking_recall_report.tsv
#     stage4_summary.json
#     stage4_blocking.duckdb
#
# ============================================================


# ============================================================
# 0. INSTALL
# ============================================================

!pip -q install duckdb pandas numpy scikit-learn pyarrow psutil


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import gc
import csv
import json
import time
import shutil
import warnings

import duckdb
import numpy as np
import pandas as pd
import psutil

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


# ============================================================
# 2. EXACT DATASET ROOT
# ============================================================

DATASET_ROOT = (
    "/content/drive/MyDrive/Colab Notebooks/dataset"
)


# ============================================================
# 3. EXACT STAGE-3 TRAIN PATHS
# ============================================================

S1_TRAIN = (
    "/content/drive/MyDrive/Colab Notebooks/dataset/"
    "stage 3/train/"
    "Copy of train_source1_stage3_normalized.tsv"
)

S2_TRAIN = (
    "/content/drive/MyDrive/Colab Notebooks/dataset/"
    "stage 3/train/"
    "Copy of train_source2_stage3_normalized.tsv"
)

S3_TRAIN = (
    "/content/drive/MyDrive/Colab Notebooks/dataset/"
    "stage 3/train/"
    "Copy of train_source3_stage3_normalized.tsv"
)


# ============================================================
# 4. EXACT STAGE-3 TEST PATHS
# ============================================================

S1_TEST = (
    "/content/drive/MyDrive/Colab Notebooks/dataset/"
    "stage 3/test/"
    "Copy of test_source1_stage3_normalized.tsv"
)

S2_TEST = (
    "/content/drive/MyDrive/Colab Notebooks/dataset/"
    "stage 3/test/"
    "Copy of test_source2_stage3_normalized.tsv"
)

S3_TEST = (
    "/content/drive/MyDrive/Colab Notebooks/dataset/"
    "stage 3/test/"
    "Copy of Copy of test_source3_stage3_normalized.tsv"
)


# ============================================================
# 5. EXACT GROUND TRUTH
# ============================================================

GROUND_TRUTH_PATH = (
    "/content/drive/MyDrive/Colab Notebooks/dataset/"
    "stage 3/train/"
    "Copy of Copy of train_ground_truth.tsv"
)


# ============================================================
# 6. NEW STAGE-4 REVISED OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    "/content/drive/MyDrive/Colab Notebooks/dataset/"
    "stage_4_revised"
)


# ============================================================
# 7. LOCAL DUCKDB WORKSPACE
# ============================================================
#
# NEVER put DuckDB temp files on Google Drive.
#
# ============================================================

LOCAL_WORK_DIR = (
    "/content/amazon_ml_stage4_revised_work"
)

LOCAL_TEMP_DIR = os.path.join(
    LOCAL_WORK_DIR,
    "duckdb_temp"
)

LOCAL_DB_PATH = os.path.join(
    LOCAL_WORK_DIR,
    "stage4_blocking.duckdb"
)


os.makedirs(
    LOCAL_WORK_DIR,
    exist_ok=True
)

os.makedirs(
    LOCAL_TEMP_DIR,
    exist_ok=True
)


# ============================================================
# 8. RESET ONLY THE NEW STAGE-4 REVISED OUTPUT
# ============================================================

RESET_OUTPUT = True


if (

    RESET_OUTPUT

    and

    os.path.exists(
        OUTPUT_DIR
    )

):

    print(
        "Removing previous stage_4_revised output..."
    )

    shutil.rmtree(
        OUTPUT_DIR
    )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 9. SETTINGS
# ============================================================

MAX_BLOCK_SIZE = 1000


# TF-IDF
TFIDF_TOP_K = 15

TFIDF_MAX_FEATURES = 50_000

TFIDF_BUCKET_MAX_ROWS = 120_000

TFIDF_MIN_DF = 1


# Apply TF-IDF to S1 rows with fewer deterministic candidates.
MAX_DETERMINISTIC_CANDIDATES = 30


# TF-IDF name representation.
CHAR_NGRAM_RANGE = (
    3,
    5
)


# ============================================================
# 10. CPU / RAM
# ============================================================

CPU_CORES = (
    os.cpu_count()
    or
    2
)


DUCKDB_THREADS = max(
    2,
    min(
        4,
        CPU_CORES
    )
)


RAM_GB = (
    psutil.virtual_memory().total
    /
    (1024 ** 3)
)


DUCKDB_MEMORY_GB = max(
    4,
    int(
        RAM_GB * 0.65
    )
)


print(
    "\n"
    "============================================================"
)

print(
    "AMAZON ML CHALLENGE 2026"
)

print(
    "STAGE 4 — BLOCKING / CANDIDATE GENERATION"
)

print(
    "REVISED FINAL VERSION"
)

print(
    "============================================================"
)

print(
    "\nCPU cores:",
    CPU_CORES
)

print(
    "DuckDB threads:",
    DUCKDB_THREADS
)

print(
    "RAM:",
    f"{RAM_GB:.1f} GB"
)

print(
    "DuckDB memory:",
    f"{DUCKDB_MEMORY_GB} GB"
)

print(
    "MAX BLOCK SIZE:",
    MAX_BLOCK_SIZE
)

print(
    "TF-IDF TOP K:",
    TFIDF_TOP_K
)

print(
    "MAX deterministic candidates:",
    MAX_DETERMINISTIC_CANDIDATES
)


# ============================================================
# 11. FILE CHECK
# ============================================================

required_files = [

    S1_TRAIN,
    S2_TRAIN,
    S3_TRAIN,

    S1_TEST,
    S2_TEST,
    S3_TEST,

    GROUND_TRUTH_PATH

]


missing_files = [

    path

    for path

    in required_files

    if not os.path.exists(path)

]


if missing_files:

    print(
        "\nMissing files:"
    )

    for path in missing_files:

        print(
            " -",
            path
        )

    raise FileNotFoundError(
        "Required Stage-3 / ground-truth files are missing."
    )


print(
    "\nAll Stage-3 and ground-truth files found."
)

print(
    "Stage-3 row-count completeness check: DISABLED"
)


# ============================================================
# 12. SQL HELPERS
# ============================================================

def sql_ident(
    name
):

    return (

        '"'
        +

        str(name).replace(
            '"',
            '""'
        )

        +

        '"'

    )


def escape_sql_string(
    value
):

    return (

        str(value).replace(
            "'",
            "''"
        )

    )


def timed_execute(
    connection,
    sql,
    label
):

    print(
        f"\n[START] {label}"
    )

    start = time.time()


    result = connection.execute(
        sql
    )


    print(
        f"[DONE] {label}"
        f" ({(time.time() - start) / 60:.1f} min)"
    )


    return result


# ============================================================
# 13. READ HEADER
# ============================================================

def read_header(
    path
):

    with open(

        path,

        "r",

        encoding="utf-8-sig",

        newline=""

    ) as f:

        line = f.readline()


    if not line:

        raise ValueError(
            f"Empty file: {path}"
        )


    return [

        item.strip()

        for item

        in (

            line

            .rstrip(
                "\r\n"
            )

            .split(
                "\t"
            )

        )

    ]


# ============================================================
# 14. BUILD DUCKDB CSV SCHEMA
# ============================================================

def build_csv_schema(
    header
):

    parts = []


    for column in header:

        parts.append(

            "'"
            +

            str(column).replace(
                "'",
                "''"
            )

            +

            "': 'VARCHAR'"

        )


    return (

        "{"

        +

        ", ".join(parts)

        +

        "}"

    )


# ============================================================
# 15. CONNECT LOCAL DUCKDB
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "CONNECTING TO LOCAL DUCKDB"
)

print(
    "============================================================"
)


if os.path.exists(
    LOCAL_DB_PATH
):

    try:

        os.remove(
            LOCAL_DB_PATH
        )

    except Exception:

        pass


con = duckdb.connect(
    LOCAL_DB_PATH
)


con.execute(
    f"PRAGMA threads={DUCKDB_THREADS}"
)


con.execute(
    f"PRAGMA memory_limit='{DUCKDB_MEMORY_GB}GB'"
)


con.execute(
    "SET preserve_insertion_order=false"
)


con.execute(

    f"""
    SET temp_directory=
    '{escape_sql_string(LOCAL_TEMP_DIR)}'
    """

)


con.execute(
    "PRAGMA enable_progress_bar=false"
)


print(
    "DuckDB:",
    LOCAL_DB_PATH
)

print(
    "Temp:",
    LOCAL_TEMP_DIR
)


# ============================================================
# 16. LOAD A TSV INTO DUCKDB
# ============================================================

def load_tsv(
    table_name,
    path
):

    print(
        "\n"
        "------------------------------------------------------------"
    )

    print(
        "LOADING",
        table_name
    )

    print(
        path
    )


    header = read_header(
        path
    )


    schema = build_csv_schema(
        header
    )


    sql = f"""

    CREATE OR REPLACE TABLE
        {sql_ident(table_name)}

    AS

    SELECT *

    FROM read_csv(

        '{escape_sql_string(path)}',

        auto_detect=false,

        delim='\\t',

        header=true,

        columns={schema},

        quote='"',

        escape='"',

        all_varchar=true,

        null_padding=true,

        strict_mode=true,

        ignore_errors=false,

        max_line_size=10000000

    );

    """


    start = time.time()


    con.execute(
        sql
    )


    count = con.execute(

        f"""
        SELECT COUNT(*)
        FROM {sql_ident(table_name)}
        """

    ).fetchone()[0]


    print(
        f"{table_name}: {count:,} rows"
    )

    print(
        "Time:",
        f"{(time.time() - start) / 60:.1f} min"
    )


    return header, count


# ============================================================
# 17. LOAD ALL SIX STAGE-3 DATASETS + GT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "LOADING STAGE-3 DATA"
)

print(
    "============================================================"
)


train_s1_header, train_s1_count = load_tsv(
    "train_s1_raw",
    S1_TRAIN
)

train_s2_header, train_s2_count = load_tsv(
    "train_s2_raw",
    S2_TRAIN
)

train_s3_header, train_s3_count = load_tsv(
    "train_s3_raw",
    S3_TRAIN
)


test_s1_header, test_s1_count = load_tsv(
    "test_s1_raw",
    S1_TEST
)

test_s2_header, test_s2_count = load_tsv(
    "test_s2_raw",
    S2_TEST
)

test_s3_header, test_s3_count = load_tsv(
    "test_s3_raw",
    S3_TEST
)


gt_header, gt_count = load_tsv(
    "train_ground_truth_raw",
    GROUND_TRUTH_PATH
)


# ============================================================
# 18. REQUIRED STAGE-3 COLUMNS
# ============================================================

REQUIRED_COLUMNS = [

    "entity_id",

    "business_name_normalized_original",

    "business_name_normalized_roman",

    "business_name_roman_phonetic",

    "business_address_normalized_original",

    "business_address_normalized_roman",

    "address_house_number",

    "address_postal_code",

    "address_locality",

    "address_city",

    "address_state_region",

    "country"

]


# ============================================================
# 19. VERIFY STAGE-3 COLUMNS
# ============================================================

def verify_required_columns(
    header,
    dataset_name
):

    missing = [

        column

        for column in REQUIRED_COLUMNS

        if column not in header

    ]


    if missing:

        print(
            f"\n{dataset_name} missing:"
        )

        for column in missing:

            print(
                " -",
                column
            )

        raise ValueError(

            f"{dataset_name} does not contain "
            f"the required Stage-3 columns."

        )


    print(
        f"{dataset_name}: required columns PASS"
    )


verify_required_columns(
    train_s1_header,
    "Train S1"
)

verify_required_columns(
    train_s2_header,
    "Train S2"
)

verify_required_columns(
    train_s3_header,
    "Train S3"
)

verify_required_columns(
    test_s1_header,
    "Test S1"
)

verify_required_columns(
    test_s2_header,
    "Test S2"
)

verify_required_columns(
    test_s3_header,
    "Test S3"
)


# ============================================================
# 20. CREATE FEATURE TABLE
# ============================================================

def create_feature_table(

    raw_table,
    output_table

):

    timed_execute(

        con,

        f"""

        CREATE OR REPLACE TABLE
            {sql_ident(output_table)}

        AS

        SELECT

            CAST(
                entity_id
                AS VARCHAR
            )
            AS entity_id,


            lower(
                trim(
                    coalesce(
                        business_name_normalized_original,
                        ''
                    )
                )
            )
            AS name_key_base,


            lower(
                trim(
                    coalesce(
                        business_name_normalized_roman,
                        ''
                    )
                )
            )
            AS roman_name_key_base,


            lower(
                trim(
                    coalesce(
                        business_name_roman_phonetic,
                        ''
                    )
                )
            )
            AS phonetic_key_base,


            lower(
                trim(
                    coalesce(
                        country,
                        ''
                    )
                )
            )
            AS country_key,


            lower(
                trim(
                    coalesce(
                        address_postal_code,
                        ''
                    )
                )
            )
            AS postal_key,


            lower(
                trim(
                    coalesce(
                        address_house_number,
                        ''
                    )
                )
            )
            AS house_key,


            lower(
                trim(
                    coalesce(
                        address_locality,
                        ''
                    )
                )
            )
            AS locality_key,


            lower(
                trim(
                    coalesce(
                        address_city,
                        ''
                    )
                )
            )
            AS city_key,


            lower(
                trim(
                    coalesce(
                        address_state_region,
                        ''
                    )
                )
            )
            AS state_key,


            lower(
                trim(
                    coalesce(
                        business_address_normalized_original,
                        ''
                    )
                )
            )
            AS address_key_base,


            lower(
                trim(
                    coalesce(
                        business_address_normalized_roman,
                        ''
                    )
                )
            )
            AS roman_address_key_base,


            CASE

                WHEN

                    lower(
                        trim(
                            coalesce(
                                business_name_normalized_original,
                                ''
                            )
                        )
                    ) <> ''

                    AND

                    lower(
                        trim(
                            coalesce(
                                country,
                                ''
                            )
                        )
                    ) <> ''

                THEN

                    lower(
                        trim(
                            coalesce(
                                business_name_normalized_original,
                                ''
                            )
                        )
                    )
                    || '¦' ||
                    lower(
                        trim(
                            coalesce(
                                country,
                                ''
                            )
                        )
                    )

                ELSE ''

            END
            AS name_country_key,


            CASE

                WHEN

                    lower(
                        trim(
                            coalesce(
                                business_name_normalized_original,
                                ''
                            )
                        )
                    ) <> ''

                    AND

                    lower(
                        trim(
                            coalesce(
                                address_postal_code,
                                ''
                            )
                        )
                    ) <> ''

                THEN

                    lower(
                        trim(
                            coalesce(
                                business_name_normalized_original,
                                ''
                            )
                        )
                    )
                    || '¦' ||
                    lower(
                        trim(
                            coalesce(
                                address_postal_code,
                                ''
                            )
                        )
                    )

                ELSE ''

            END
            AS name_postal_key,


            CASE

                WHEN

                    lower(
                        trim(
                            coalesce(
                                address_house_number,
                                ''
                            )
                        )
                    ) <> ''

                    AND

                    lower(
                        trim(
                            coalesce(
                                address_locality,
                                ''
                            )
                        )
                    ) <> ''

                THEN

                    lower(
                        trim(
                            coalesce(
                                address_house_number,
                                ''
                            )
                        )
                    )
                    || '¦' ||
                    lower(
                        trim(
                            coalesce(
                                address_locality,
                                ''
                            )
                        )
                    )

                ELSE ''

            END
            AS house_locality_key,


            CASE

                WHEN

                    lower(
                        trim(
                            coalesce(
                                address_house_number,
                                ''
                            )
                        )
                    ) <> ''

                    AND

                    lower(
                        trim(
                            coalesce(
                                address_city,
                                ''
                            )
                        )
                    ) <> ''

                THEN

                    lower(
                        trim(
                            coalesce(
                                address_house_number,
                                ''
                            )
                        )
                    )
                    || '¦' ||
                    lower(
                        trim(
                            coalesce(
                                address_city,
                                ''
                            )
                        )
                    )

                ELSE ''

            END
            AS house_city_key,


            regexp_extract(

                lower(
                    trim(
                        coalesce(
                            business_name_normalized_roman,
                            ''
                        )
                    )
                ),

                '([a-z0-9]{{2}})',

                1

            )
            AS tfidf_prefix2

        FROM
            {sql_ident(raw_table)};

        """,

        f"Create {output_table}"

    )


# ============================================================
# 21. CREATE FEATURE TABLES
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "CREATING FEATURE TABLES"
)

print(
    "============================================================"
)


create_feature_table(
    "train_s1_raw",
    "f_train_s1"
)

create_feature_table(
    "train_s2_raw",
    "f_train_s2"
)

create_feature_table(
    "train_s3_raw",
    "f_train_s3"
)


create_feature_table(
    "test_s1_raw",
    "f_test_s1"
)

create_feature_table(
    "test_s2_raw",
    "f_test_s2"
)

create_feature_table(
    "test_s3_raw",
    "f_test_s3"
)


# ============================================================
# 22. SHOW COUNTS
# ============================================================

print(
    "\n"
    "========== FEATURE TABLE COUNTS =========="
)


for table in [

    "f_train_s1",
    "f_train_s2",
    "f_train_s3",
    "f_test_s1",
    "f_test_s2",
    "f_test_s3"

]:

    count = con.execute(

        f"""
        SELECT COUNT(*)
        FROM {table}
        """

    ).fetchone()[0]


    print(
        f"{table}: {count:,}"
    )


# ============================================================
# 23. CREATE CANDIDATE LONG TABLES
# ============================================================

con.execute(
    """
    CREATE OR REPLACE TABLE train_candidate_long (
        source1_entity_id VARCHAR,
        candidate_entity_id VARCHAR,
        source_pair VARCHAR,
        strategy VARCHAR
    );
    """
)


con.execute(
    """
    CREATE OR REPLACE TABLE test_candidate_long (
        source1_entity_id VARCHAR,
        candidate_entity_id VARCHAR,
        source_pair VARCHAR,
        strategy VARCHAR
    );
    """
)


# ============================================================
# 24. EXACT BLOCK FUNCTION
# ============================================================

def add_exact_block(

    s1_table,
    sx_table,
    target_table,
    source_pair,
    key_column,
    strategy

):

    # --------------------------------------------------------
    # Create eligible keys on S1.
    # --------------------------------------------------------

    sql = f"""

    INSERT INTO
        {target_table}

    SELECT

        a.entity_id
        AS source1_entity_id,

        b.entity_id
        AS candidate_entity_id,

        '{source_pair}'
        AS source_pair,

        '{strategy}'
        AS strategy

    FROM
        {s1_table} a

    INNER JOIN
        {sx_table} b

        ON

        a.{sql_ident(key_column)}
        =
        b.{sql_ident(key_column)}

    INNER JOIN

    (

        SELECT

            {sql_ident(key_column)}
            AS key_value

        FROM
            {s1_table}

        WHERE

            {sql_ident(key_column)}
            <> ''

        GROUP BY

            {sql_ident(key_column)}

        HAVING

            COUNT(*) <= {MAX_BLOCK_SIZE}

    ) ak

        ON

        ak.key_value
        =
        a.{sql_ident(key_column)}

    INNER JOIN

    (

        SELECT

            {sql_ident(key_column)}
            AS key_value

        FROM
            {sx_table}

        WHERE

            {sql_ident(key_column)}
            <> ''

        GROUP BY

            {sql_ident(key_column)}

        HAVING

            COUNT(*) <= {MAX_BLOCK_SIZE}

    ) bk

        ON

        bk.key_value
        =
        b.{sql_ident(key_column)};

    """


    timed_execute(

        con,

        sql,

        f"{strategy} [{source_pair}]"

    )


# ============================================================
# 25. HOUSE + LOCATION BLOCK
# ============================================================

def add_house_location_block(

    s1_table,
    sx_table,
    target_table,
    source_pair

):

    # --------------------------------------------------------
    # House + locality
    # --------------------------------------------------------

    add_exact_block(

        s1_table,
        sx_table,
        target_table,
        source_pair,
        "house_locality_key",
        "house_locality"

    )


    # --------------------------------------------------------
    # House + city
    # --------------------------------------------------------

    add_exact_block(

        s1_table,
        sx_table,
        target_table,
        source_pair,
        "house_city_key",
        "house_city"

    )


# ============================================================
# 26. RUN DETERMINISTIC TRAIN BLOCKING
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "TRAIN DETERMINISTIC BLOCKING"
)

print(
    "============================================================"
)


train_source_pairs = [

    (
        "f_train_s2",
        "s1_s2"
    ),

    (
        "f_train_s3",
        "s1_s3"
    )

]


for sx_table, source_pair in train_source_pairs:

    # 1. Exact normalized name
    add_exact_block(

        "f_train_s1",
        sx_table,
        "train_candidate_long",
        source_pair,
        "name_key_base",
        "exact_name"

    )


    # 2. Name + country
    add_exact_block(

        "f_train_s1",
        sx_table,
        "train_candidate_long",
        source_pair,
        "name_country_key",
        "name_country"

    )


    # 3. Name + postal
    add_exact_block(

        "f_train_s1",
        sx_table,
        "train_candidate_long",
        source_pair,
        "name_postal_key",
        "name_postal"

    )


    # 4. House/location
    add_house_location_block(

        "f_train_s1",
        sx_table,
        "train_candidate_long",
        source_pair

    )


    # 5. Phonetic
    add_exact_block(

        "f_train_s1",
        sx_table,
        "train_candidate_long",
        source_pair,
        "phonetic_key_base",
        "phonetic_name"

    )


    # 6. Exact Romanized name
    add_exact_block(

        "f_train_s1",
        sx_table,
        "train_candidate_long",
        source_pair,
        "roman_name_key_base",
        "roman_exact_name"

    )


# ============================================================
# 27. RUN DETERMINISTIC TEST BLOCKING
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "TEST DETERMINISTIC BLOCKING"
)

print(
    "============================================================"
)


test_source_pairs = [

    (
        "f_test_s2",
        "s1_s2"
    ),

    (
        "f_test_s3",
        "s1_s3"
    )

]


for sx_table, source_pair in test_source_pairs:

    add_exact_block(

        "f_test_s1",
        sx_table,
        "test_candidate_long",
        source_pair,
        "name_key_base",
        "exact_name"

    )


    add_exact_block(

        "f_test_s1",
        sx_table,
        "test_candidate_long",
        source_pair,
        "name_country_key",
        "name_country"

    )


    add_exact_block(

        "f_test_s1",
        sx_table,
        "test_candidate_long",
        source_pair,
        "name_postal_key",
        "name_postal"

    )


    add_house_location_block(

        "f_test_s1",
        sx_table,
        "test_candidate_long",
        source_pair

    )


    add_exact_block(

        "f_test_s1",
        sx_table,
        "test_candidate_long",
        source_pair,
        "phonetic_key_base",
        "phonetic_name"

    )


    add_exact_block(

        "f_test_s1",
        sx_table,
        "test_candidate_long",
        source_pair,
        "roman_name_key_base",
        "roman_exact_name"

    )


# ============================================================
# 28. REPORT DETERMINISTIC CANDIDATES
# ============================================================

print(
    "\n"
    "========== DETERMINISTIC CANDIDATE COUNTS =========="
)


for table in [

    "train_candidate_long",
    "test_candidate_long"

]:

    count = con.execute(

        f"""
        SELECT COUNT(*)
        FROM {table}
        """

    ).fetchone()[0]


    unique_count = con.execute(

        f"""
        SELECT COUNT(*)

        FROM
        (
            SELECT DISTINCT

                source1_entity_id,
                candidate_entity_id,
                source_pair

            FROM {table}
        );

        """

    ).fetchone()[0]


    print(
        f"{table}: raw={count:,} "
        f"unique={unique_count:,}"
    )


# ============================================================
# 29. CREATE UNIQUE DETERMINISTIC PAIR TABLES
# ============================================================

timed_execute(

    con,

    """

    CREATE OR REPLACE TABLE
        train_candidate_unique

    AS

    SELECT DISTINCT

        source1_entity_id,

        candidate_entity_id,

        source_pair

    FROM
        train_candidate_long;

    """,

    "Train deterministic deduplication"

)


timed_execute(

    con,

    """

    CREATE OR REPLACE TABLE
        test_candidate_unique

    AS

    SELECT DISTINCT

        source1_entity_id,

        candidate_entity_id,

        source_pair

    FROM
        test_candidate_long;

    """,

    "Test deterministic deduplication"

)


# ============================================================
# 30. GET UNRESOLVED S1
# ============================================================

def get_unresolved_s1(

    s1_table,
    candidate_unique_table

):

    return con.execute(

        f"""

        SELECT

            a.entity_id,

            a.roman_name_key_base,

            a.tfidf_prefix2,

            a.country_key

        FROM
            {s1_table} a

        LEFT JOIN

        (

            SELECT

                source1_entity_id,

                COUNT(*) AS candidate_count

            FROM
                {candidate_unique_table}

            GROUP BY
                source1_entity_id

        ) c

            ON

            c.source1_entity_id
            =
            a.entity_id

        WHERE

            COALESCE(
                c.candidate_count,
                0
            )
            <
            {MAX_DETERMINISTIC_CANDIDATES}

            AND

            a.roman_name_key_base
            <> '';

        """

    ).df()


# ============================================================
# 31. TF-IDF BLOCK
# ============================================================

def run_tfidf_block(

    s1_table,
    sx_table,
    candidate_unique_table,
    candidate_long_table,
    source_pair

):

    print(
        "\n"
        "============================================================"
    )

    print(
        f"TF-IDF BLOCKING — {source_pair}"
    )

    print(
        "============================================================"
    )


    unresolved = get_unresolved_s1(

        s1_table,

        candidate_unique_table

    )


    print(

        "S1 records below deterministic threshold:",

        f"{len(unresolved):,}"

    )


    if unresolved.empty:

        print(
            "No unresolved S1 records."
        )

        return


    # --------------------------------------------------------
    # Bucket
    # --------------------------------------------------------

    unresolved["bucket"] = (

        unresolved[
            "country_key"
        ]
        .fillna("")
        .astype(str)

        +

        "¦"

        +

        unresolved[
            "tfidf_prefix2"
        ]
        .fillna("")
        .astype(str)

    )


    bucket_values = (

        unresolved[
            "bucket"
        ]

        .value_counts()

        .index

        .tolist()

    )


    print(
        "TF-IDF buckets:",
        f"{len(bucket_values):,}"
    )


    inserted = 0

    skipped_large = 0

    empty_source = 0


    start_total = time.time()


    # ========================================================
    # BUCKET LOOP
    # ========================================================

    for bucket_number, bucket in enumerate(

        bucket_values,

        start=1

    ):

        s1_bucket = unresolved.loc[

            unresolved[
                "bucket"
            ]
            ==
            bucket

        ]


        if s1_bucket.empty:

            continue


        parts = bucket.split(
            "¦",
            1
        )


        if len(parts) != 2:

            continue


        country_value = parts[0]

        prefix_value = parts[1]


        # ----------------------------------------------------
        # Get candidate source records
        # ----------------------------------------------------

        source_df = con.execute(

            f"""

            SELECT

                entity_id,

                roman_name_key_base

            FROM
                {sx_table}

            WHERE

                roman_name_key_base
                <> ''

                AND

                country_key = ?

                AND

                tfidf_prefix2 = ?

            """,

            [

                country_value,

                prefix_value

            ]

        ).df()


        if source_df.empty:

            empty_source += 1

            continue


        if (

            len(source_df)
            >
            TFIDF_BUCKET_MAX_ROWS

        ):

            skipped_large += 1

            continue


        # ----------------------------------------------------
        # Build corpus
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # TF-IDF vectorizer
        # ----------------------------------------------------

        try:

            vectorizer = TfidfVectorizer(

                analyzer="char",

                ngram_range=CHAR_NGRAM_RANGE,

                min_df=TFIDF_MIN_DF,

                max_features=TFIDF_MAX_FEATURES,

                lowercase=False,

                sublinear_tf=True,

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

                len(source_df)

            )


            nn = NearestNeighbors(

                n_neighbors=k,

                metric="cosine",

                algorithm="brute",

                n_jobs=-1

            )


            nn.fit(
                X
            )


            distances, indices = nn.kneighbors(

                Q,

                return_distance=True

            )


            rows = []


            for i in range(

                len(s1_bucket)

            ):

                source1_id = str(

                    s1_bucket.iloc[i][
                        "entity_id"
                    ]

                )


                for j in range(

                    indices.shape[1]

                ):

                    source_idx = int(

                        indices[
                            i,
                            j
                        ]

                    )


                    candidate_id = str(

                        source_df.iloc[
                            source_idx
                        ][
                            "entity_id"
                        ]

                    )


                    rows.append(

                        (

                            source1_id,

                            candidate_id,

                            source_pair,

                            "tfidf_char_topk"

                        )

                    )


            if rows:

                temp_df = pd.DataFrame(

                    rows,

                    columns=[

                        "source1_entity_id",

                        "candidate_entity_id",

                        "source_pair",

                        "strategy"

                    ]

                )


                con.register(

                    "tmp_tfidf_rows",

                    temp_df

                )


                con.execute(

                    f"""

                    INSERT INTO
                        {candidate_long_table}

                    SELECT *

                    FROM
                        tmp_tfidf_rows;

                    """

                )


                con.unregister(
                    "tmp_tfidf_rows"
                )


                inserted += len(
                    rows
                )


        except Exception as error:

            print(

                f"\nTF-IDF bucket error "
                f"{bucket_number}:",
                error

            )


        del source_df

        gc.collect()


        if (

            bucket_number % 100 == 0

            or

            bucket_number == len(bucket_values)

        ):

            elapsed = (

                time.time()
                -
                start_total

            ) / 60


            print(

                f"Bucket "
                f"{bucket_number:,}/"
                f"{len(bucket_values):,} | "

                f"inserted="
                f"{inserted:,} | "

                f"oversized="
                f"{skipped_large:,} | "

                f"time="
                f"{elapsed:.1f} min"

            )


    print(

        "\nTF-IDF complete."
    )


    print(

        "Inserted TF-IDF candidates:",
        f"{inserted:,}"

    )


    print(

        "Oversized buckets skipped:",
        f"{skipped_large:,}"

    )


    print(

        "Empty source buckets:",
        f"{empty_source:,}"

    )


# ============================================================
# 32. TRAIN TF-IDF
# ============================================================

run_tfidf_block(

    "f_train_s1",

    "f_train_s2",

    "train_candidate_unique",

    "train_candidate_long",

    "s1_s2"

)


# IMPORTANT:
# After adding TF-IDF to long, refresh unique table.

timed_execute(

    con,

    """

    CREATE OR REPLACE TABLE
        train_candidate_unique

    AS

    SELECT DISTINCT

        source1_entity_id,

        candidate_entity_id,

        source_pair

    FROM
        train_candidate_long;

    """,

    "Refresh train unique candidates after S2 TF-IDF"

)


run_tfidf_block(

    "f_train_s1",

    "f_train_s3",

    "train_candidate_unique",

    "train_candidate_long",

    "s1_s3"

)


# ============================================================
# 33. TEST TF-IDF
# ============================================================

run_tfidf_block(

    "f_test_s1",

    "f_test_s2",

    "test_candidate_unique",

    "test_candidate_long",

    "s1_s2"

)


timed_execute(

    con,

    """

    CREATE OR REPLACE TABLE
        test_candidate_unique

    AS

    SELECT DISTINCT

        source1_entity_id,

        candidate_entity_id,

        source_pair

    FROM
        test_candidate_long;

    """,

    "Refresh test unique candidates after S2 TF-IDF"

)


run_tfidf_block(

    "f_test_s1",

    "f_test_s3",

    "test_candidate_unique",

    "test_candidate_long",

    "s1_s3"

)


# ============================================================
# 34. FINAL DEDUPLICATION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "FINAL CANDIDATE DEDUPLICATION"
)

print(
    "============================================================"
)


timed_execute(

    con,

    """

    CREATE OR REPLACE TABLE
        train_candidate_unique

    AS

    SELECT DISTINCT

        source1_entity_id,

        candidate_entity_id,

        source_pair

    FROM
        train_candidate_long;

    """,

    "Final train candidate deduplication"

)


timed_execute(

    con,

    """

    CREATE OR REPLACE TABLE
        test_candidate_unique

    AS

    SELECT DISTINCT

        source1_entity_id,

        candidate_entity_id,

        source_pair

    FROM
        test_candidate_long;

    """,

    "Final test candidate deduplication"

)


train_long_count = con.execute(

    """
    SELECT COUNT(*)
    FROM train_candidate_long
    """

).fetchone()[0]


train_unique_count = con.execute(

    """
    SELECT COUNT(*)
    FROM train_candidate_unique
    """

).fetchone()[0]


test_long_count = con.execute(

    """
    SELECT COUNT(*)
    FROM test_candidate_long
    """

).fetchone()[0]


test_unique_count = con.execute(

    """
    SELECT COUNT(*)
    FROM test_candidate_unique
    """

).fetchone()[0]


print(
    "\nTrain long:",
    f"{train_long_count:,}"
)

print(
    "Train unique:",
    f"{train_unique_count:,}"
)

print(
    "Test long:",
    f"{test_long_count:,}"
)

print(
    "Test unique:",
    f"{test_unique_count:,}"
)


# ============================================================
# 35. WRITE TSV STREAMING
# ============================================================
#
# This is the critical fix.
#
# We DO NOT use the previous malformed writer.
#
# Python's csv.writer with delimiter="\t" writes a REAL TAB.
#
# ============================================================

def write_candidate_pairs_tsv(

    dataset_name,
    s1_table,
    candidate_table,
    output_path

):

    print(
        "\n"
        "------------------------------------------------------------"
    )

    print(
        f"WRITING {dataset_name.upper()} CANDIDATE FILE"
    )

    print(
        output_path
    )


    query = f"""

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


    FROM
        {s1_table} a


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


    start = time.time()


    cursor = con.execute(
        query
    )


    row_count = 0

    rows_with_candidates = 0


    # --------------------------------------------------------
    # IMPORTANT:
    # newline="" and delimiter="\t"
    # --------------------------------------------------------

    with open(

        output_path,

        "w",

        encoding="utf-8",

        newline=""

    ) as output_file:

        writer = csv.writer(

            output_file,

            delimiter="\t",

            quotechar='"',

            quoting=csv.QUOTE_MINIMAL,

            lineterminator="\n"

        )


        writer.writerow(

            [

                "source1_entity_id",

                "candidate_entity_ids"

            ]

        )


        while True:

            batch = cursor.fetchmany(
                10_000
            )


            if not batch:

                break


            for source1_id, candidate_ids in batch:

                row_count += 1


                if (

                    candidate_ids is not None

                    and

                    str(
                        candidate_ids
                    ).strip()
                    != ""

                ):

                    rows_with_candidates += 1


                writer.writerow(

                    [

                        source1_id,

                        candidate_ids
                        if candidate_ids is not None
                        else ""

                    ]

                )


            if (

                row_count % 100_000
                ==
                0

            ):

                print(

                    f"Written "
                    f"{row_count:,} rows..."

                )


    elapsed = (

        time.time()
        -
        start

    ) / 60


    print(

        f"{dataset_name}:"
        f" {row_count:,} rows"

    )

    print(

        f"{dataset_name}:"
        f" {rows_with_candidates:,} "
        f"rows with candidates"

    )

    print(
        "Time:",
        f"{elapsed:.1f} min"
    )


    return {

        "rows":
            row_count,

        "rows_with_candidates":
            rows_with_candidates

    }


# ============================================================
# 36. OUTPUT PATHS
# ============================================================

TRAIN_CANDIDATE_OUTPUT = os.path.join(

    OUTPUT_DIR,

    "train_candidate_pairs.tsv"

)


TEST_CANDIDATE_OUTPUT = os.path.join(

    OUTPUT_DIR,

    "test_candidate_pairs.tsv"

)


FINAL_CANDIDATE_OUTPUT = os.path.join(

    OUTPUT_DIR,

    "candidate_pairs.tsv"

)


# ============================================================
# 37. WRITE TRAIN CANDIDATES
# ============================================================

train_output_summary = write_candidate_pairs_tsv(

    "train",

    "f_train_s1",

    "train_candidate_unique",

    TRAIN_CANDIDATE_OUTPUT

)


# ============================================================
# 38. WRITE TEST CANDIDATES
# ============================================================

test_output_summary = write_candidate_pairs_tsv(

    "test",

    "f_test_s1",

    "test_candidate_unique",

    TEST_CANDIDATE_OUTPUT

)


# ============================================================
# 39. CREATE FINAL candidate_pairs.tsv
# ============================================================
#
# This is simply the TEST candidate set.
#
# Write it independently again to guarantee a valid TSV.
#
# ============================================================

final_output_summary = write_candidate_pairs_tsv(

    "final test",

    "f_test_s1",

    "test_candidate_unique",

    FINAL_CANDIDATE_OUTPUT

)


# ============================================================
# 40. VERIFY FILE BYTES
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "VERIFYING FINAL TSV FILES"
)

print(
    "============================================================"
)


def verify_candidate_file(
    path,
    expected_rows
):

    print(
        "\nChecking:",
        path
    )


    with open(

        path,

        "rb"

    ) as f:

        header = f.readline()

        first_data = f.readline()


    print(
        "Header bytes:",
        repr(header)
    )


    print(
        "First data line:",
        repr(first_data[:500])
    )


    expected_header = (
        b"source1_entity_id\tcandidate_entity_ids\n"
    )


    if header != expected_header:

        raise RuntimeError(

            "INVALID CANDIDATE HEADER.\n"
            f"Expected: {repr(expected_header)}\n"
            f"Actual: {repr(header)}"

        )


    row_count = (

        sum(

            1

            for _

            in open(

                path,

                "r",

                encoding="utf-8"

            )

        )

        - 1

    )


    print(
        "Rows:",
        f"{row_count:,}"
    )


    if row_count != expected_rows:

        raise RuntimeError(

            f"Candidate row count mismatch: "
            f"{row_count:,} != {expected_rows:,}"

        )


    # --------------------------------------------------------
    # Read sample using pandas.
    # --------------------------------------------------------

    sample = pd.read_csv(

        path,

        sep="\t",

        dtype=str,

        keep_default_na=False,

        nrows=5

    )


    expected_columns = [

        "source1_entity_id",

        "candidate_entity_ids"

    ]


    if list(sample.columns) != expected_columns:

        raise RuntimeError(

            "Candidate file parsed incorrectly.\n"
            f"Columns: {list(sample.columns)}"

        )


    print(
        "Parsed columns:",
        list(sample.columns)
    )


    print(
        "Sample:"
    )


    print(

        sample.to_string(
            index=False
        )

    )


    return row_count


verify_candidate_file(

    TRAIN_CANDIDATE_OUTPUT,

    train_s1_count

)


verify_candidate_file(

    TEST_CANDIDATE_OUTPUT,

    test_s1_count

)


verify_candidate_file(

    FINAL_CANDIDATE_OUTPUT,

    test_s1_count

)


# ============================================================
# 41. CANDIDATE ID VALIDATION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "VALIDATING CANDIDATE IDS"
)

print(
    "============================================================"
)


def validate_candidate_ids(

    candidate_path,
    s2_table,
    s3_table

):

    s2_count = con.execute(

        f"""
        SELECT COUNT(*)
        FROM {s2_table}
        """

    ).fetchone()[0]


    s3_count = con.execute(

        f"""
        SELECT COUNT(*)
        FROM {s3_table}
        """

    ).fetchone()[0]


    print(
        "Valid S2 records:",
        f"{s2_count:,}"
    )

    print(
        "Valid S3 records:",
        f"{s3_count:,}"
    )


    # Read candidate IDs into temporary table.
    temp_path = os.path.join(

        LOCAL_WORK_DIR,

        "candidate_validation.tsv"

    )


    with open(

        candidate_path,

        "r",

        encoding="utf-8",

        newline=""

    ) as src:

        with open(

            temp_path,

            "w",

            encoding="utf-8",

            newline=""

        ) as dst:

            reader = csv.DictReader(

                src,

                delimiter="\t"

            )


            writer = csv.writer(

                dst,

                delimiter="\t",

                lineterminator="\n"

            )


            writer.writerow(

                [

                    "source1_entity_id",

                    "candidate_entity_id"

                ]

            )


            row_count = 0


            for row in reader:

                source1_id = (

                    row[
                        "source1_entity_id"
                    ]

                    .strip()

                )


                candidate_string = (

                    row[
                        "candidate_entity_ids"
                    ]

                    .strip()

                )


                if not candidate_string:

                    continue


                seen = set()


                for candidate_id in (

                    candidate_string.split(",")

                ):

                    candidate_id = (

                        candidate_id
                        .strip()

                    )


                    if not candidate_id:
                        continue


                    if candidate_id in seen:
                        continue


                    seen.add(
                        candidate_id
                    )


                    writer.writerow(

                        [

                            source1_id,

                            candidate_id

                        ]

                    )


                    row_count += 1


    validation_con = duckdb.connect()

    validation_con.execute(

        f"""

        CREATE OR REPLACE TABLE
            candidate_validation

        AS

        SELECT *

        FROM read_csv(

            '{escape_sql_string(temp_path)}',

            delim='\\t',

            header=true,

            all_varchar=true

        );

        """

    )


    invalid_count = validation_con.execute(

        f"""

        SELECT COUNT(*)

        FROM candidate_validation c

        LEFT JOIN {s2_table} s2

            ON

            c.candidate_entity_id
            =
            s2.entity_id

        LEFT JOIN {s3_table} s3

            ON

            c.candidate_entity_id
            =
            s3.entity_id

        WHERE

            s2.entity_id IS NULL

            AND

            s3.entity_id IS NULL;

        """

    ).fetchone()[0]


    validation_con.close()


    print(
        "Expanded candidate IDs:",
        f"{row_count:,}"
    )


    print(
        "Invalid candidate IDs:",
        f"{invalid_count:,}"
    )


    os.remove(
        temp_path
    )


    if invalid_count != 0:

        raise RuntimeError(

            "Some candidate IDs do not exist "
            "in S2/S3."

        )


validate_candidate_ids(

    FINAL_CANDIDATE_OUTPUT,

    "f_test_s2",

    "f_test_s3"

)


# ============================================================
# 42. STRATEGY AUDIT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "STRATEGY AUDIT"
)

print(
    "============================================================"
)


def create_strategy_audit(

    candidate_table,
    output_path

):

    audit_df = con.execute(

        f"""

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

        FROM
            {candidate_table}

        GROUP BY

            source_pair,

            strategy

        ORDER BY

            source_pair,

            strategy;

        """

    ).df()


    audit_df.to_csv(

        output_path,

        sep="\t",

        index=False

    )


    print(
        "Saved:",
        output_path
    )


    print(

        audit_df.to_string(
            index=False
        )

    )


create_strategy_audit(

    "train_candidate_long",

    os.path.join(

        OUTPUT_DIR,

        "train_candidate_strategy_audit.tsv"

    )

)


create_strategy_audit(

    "test_candidate_long",

    os.path.join(

        OUTPUT_DIR,

        "test_candidate_strategy_audit.tsv"

    )

)


# ============================================================
# 43. CANDIDATE COUNT DISTRIBUTION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "CANDIDATE COUNT DISTRIBUTION"
)

print(
    "============================================================"
)


def create_distribution(

    s1_table,
    candidate_table,
    output_path

):

    distribution_df = con.execute(

        f"""

        SELECT

            a.entity_id
            AS source1_entity_id,

            COUNT(
                DISTINCT c.candidate_entity_id
            )
            AS candidate_count

        FROM
            {s1_table} a

        LEFT JOIN
            {candidate_table} c

            ON

            a.entity_id
            =
            c.source1_entity_id

        GROUP BY

            a.entity_id;

        """

    ).df()


    print(

        distribution_df[
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


    distribution_df.to_csv(

        output_path,

        sep="\t",

        index=False

    )


    print(
        "Saved:",
        output_path
    )


create_distribution(

    "f_train_s1",

    "train_candidate_unique",

    os.path.join(

        OUTPUT_DIR,

        "train_candidate_count_distribution.tsv"

    )

)


create_distribution(

    "f_test_s1",

    "test_candidate_unique",

    os.path.join(

        OUTPUT_DIR,

        "test_candidate_count_distribution.tsv"

    )

)


# ============================================================
# 44. BUILD GROUND-TRUTH PAIRS
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "BUILDING GROUND-TRUTH PAIRS"
)

print(
    "============================================================"
)


timed_execute(

    con,

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

        trim(x)
        AS matched_entity_id

    FROM
        train_ground_truth_raw,

        UNNEST(

            string_split(

                coalesce(
                    matched_entity_ids,
                    ''
                ),

                ','

            )

        ) AS t(x)

    WHERE

        trim(x) <> '';

    """,

    "Expand ground truth"

)


timed_execute(

    con,

    """

    CREATE OR REPLACE TABLE
        train_gt_pairs_typed

    AS

    SELECT

        g.source1_entity_id,

        g.matched_entity_id
        AS candidate_entity_id,

        CASE

            WHEN
                s2.entity_id IS NOT NULL

            THEN
                's1_s2'

            WHEN
                s3.entity_id IS NOT NULL

            THEN
                's1_s3'

            ELSE
                'unknown'

        END
        AS source_pair

    FROM
        train_gt_pairs g

    LEFT JOIN
        f_train_s2 s2

        ON

        g.matched_entity_id
        =
        s2.entity_id

    LEFT JOIN
        f_train_s3 s3

        ON

        g.matched_entity_id
        =
        s3.entity_id;

    """,

    "Type ground truth"

)


gt_pairs_count = con.execute(

    """

    SELECT COUNT(*)
    FROM train_gt_pairs_typed

    """

).fetchone()[0]


unknown_gt_count = con.execute(

    """

    SELECT COUNT(*)

    FROM train_gt_pairs_typed

    WHERE source_pair = 'unknown';

    """

).fetchone()[0]


print(
    "Ground-truth pairs:",
    f"{gt_pairs_count:,}"
)


print(
    "Unknown ground-truth IDs:",
    f"{unknown_gt_count:,}"
)


if gt_pairs_count == 0:

    raise RuntimeError(

        "GROUND TRUTH EXPANDED TO ZERO PAIRS."

    )


# ============================================================
# 45. BLOCKING RECALL
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "TRAIN BLOCKING RECALL"
)

print(
    "============================================================"
)


strategy_values = con.execute(

    """

    SELECT DISTINCT
        strategy

    FROM
        train_candidate_long

    ORDER BY
        strategy;

    """

).fetchall()


recall_records = []


for (
    strategy_tuple
) in strategy_values:

    strategy = strategy_tuple[0]


    rows = con.execute(

        """

        SELECT

            g.source_pair,

            COUNT(*) AS gt_pairs,

            COUNT(
                c.candidate_entity_id
            )
            AS recovered_pairs

        FROM
            train_gt_pairs_typed g

        LEFT JOIN
            train_candidate_long c

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
            = ?

        WHERE

            g.source_pair <> 'unknown'

        GROUP BY
            g.source_pair

        ORDER BY
            g.source_pair;

        """,

        [strategy]

    ).fetchall()


    for (

        source_pair,

        gt_pairs,

        recovered_pairs

    ) in rows:

        recall = (

            recovered_pairs
            /
            gt_pairs

            if gt_pairs

            else
            0.0

        )


        recall_records.append(

            {

                "strategy":
                    strategy,

                "source_pair":
                    source_pair,

                "gt_pairs":
                    int(gt_pairs),

                "recovered_pairs":
                    int(recovered_pairs),

                "recall":
                    float(recall)

            }

        )


# ------------------------------------------------------------
# UNION RECALL
# ------------------------------------------------------------

union_rows = con.execute(

    """

    SELECT

        g.source_pair,

        COUNT(*) AS gt_pairs,

        COUNT(
            c.candidate_entity_id
        )
        AS recovered_pairs

    FROM
        train_gt_pairs_typed g

    LEFT JOIN

    (

        SELECT DISTINCT

            source1_entity_id,

            candidate_entity_id,

            source_pair

        FROM
            train_candidate_unique

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


for (

    source_pair,

    gt_pairs,

    recovered_pairs

) in union_rows:

    recall = (

        recovered_pairs
        /
        gt_pairs

        if gt_pairs

        else
        0.0

    )


    recall_records.append(

        {

            "strategy":
                "ALL_UNION",

            "source_pair":
                source_pair,

            "gt_pairs":
                int(gt_pairs),

            "recovered_pairs":
                int(recovered_pairs),

            "recall":
                float(recall)

        }

    )


recall_df = pd.DataFrame(

    recall_records

)


recall_output_path = os.path.join(

    OUTPUT_DIR,

    "train_blocking_recall_report.tsv"

)


recall_df.to_csv(

    recall_output_path,

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
    recall_output_path
)


# ============================================================
# 46. FINAL OUTPUT STATS
# ============================================================

train_s1_with_candidates = con.execute(

    """

    SELECT COUNT(DISTINCT source1_entity_id)

    FROM train_candidate_unique;

    """

).fetchone()[0]


test_s1_with_candidates = con.execute(

    """

    SELECT COUNT(DISTINCT source1_entity_id)

    FROM test_candidate_unique;

    """

).fetchone()[0]


train_zero_candidate_s1 = (

    train_s1_count
    -
    train_s1_with_candidates

)


test_zero_candidate_s1 = (

    test_s1_count
    -
    test_s1_with_candidates

)


print(
    "\n"
    "============================================================"
)

print(
    "FINAL CANDIDATE STATISTICS"
)

print(
    "============================================================"
)


print(
    "Train S1:",
    f"{train_s1_count:,}"
)

print(
    "Train S1 with candidates:",
    f"{train_s1_with_candidates:,}"
)

print(
    "Train S1 without candidates:",
    f"{train_zero_candidate_s1:,}"
)

print(
    "Train unique candidate pairs:",
    f"{train_unique_count:,}"
)


print(
    "\nTest S1:",
    f"{test_s1_count:,}"
)

print(
    "Test S1 with candidates:",
    f"{test_s1_with_candidates:,}"
)

print(
    "Test S1 without candidates:",
    f"{test_zero_candidate_s1:,}"
)

print(
    "Test unique candidate pairs:",
    f"{test_unique_count:,}"
)


# ============================================================
# 47. FINAL SUMMARY
# ============================================================

summary = {

    "stage":
        "Stage 4 - Blocking / Candidate Generation",

    "version":
        "revised",

    "output_directory":
        OUTPUT_DIR,

    "local_duckdb":
        LOCAL_DB_PATH,

    "stage3_row_count_check":
        "DISABLED",

    "inputs":

        {

            "train_s1":
                S1_TRAIN,

            "train_s2":
                S2_TRAIN,

            "train_s3":
                S3_TRAIN,

            "test_s1":
                S1_TEST,

            "test_s2":
                S2_TEST,

            "test_s3":
                S3_TEST,

            "ground_truth":
                GROUND_TRUTH_PATH

        },

    "input_row_counts":

        {

            "train_s1":
                int(train_s1_count),

            "train_s2":
                int(train_s2_count),

            "train_s3":
                int(train_s3_count),

            "test_s1":
                int(test_s1_count),

            "test_s2":
                int(test_s2_count),

            "test_s3":
                int(test_s3_count),

            "ground_truth":
                int(gt_count)

        },

    "candidate_pairs":

        {

            "train_raw":
                int(train_long_count),

            "train_unique":
                int(train_unique_count),

            "test_raw":
                int(test_long_count),

            "test_unique":
                int(test_unique_count)

        },

    "coverage":

        {

            "train_s1_with_candidates":
                int(train_s1_with_candidates),

            "train_s1_without_candidates":
                int(train_zero_candidate_s1),

            "test_s1_with_candidates":
                int(test_s1_with_candidates),

            "test_s1_without_candidates":
                int(test_zero_candidate_s1)

        },

    "outputs":

        {

            "train_candidate_pairs":
                TRAIN_CANDIDATE_OUTPUT,

            "test_candidate_pairs":
                TEST_CANDIDATE_OUTPUT,

            "candidate_pairs":
                FINAL_CANDIDATE_OUTPUT,

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

            "train_distribution":
                os.path.join(

                    OUTPUT_DIR,

                    "train_candidate_count_distribution.tsv"

                ),

            "test_distribution":
                os.path.join(

                    OUTPUT_DIR,

                    "test_candidate_count_distribution.tsv"

                ),

            "recall_report":
                recall_output_path

        },

    "blocking_settings":

        {

            "max_block_size":
                MAX_BLOCK_SIZE,

            "tfidf_top_k":
                TFIDF_TOP_K,

            "tfidf_max_features":
                TFIDF_MAX_FEATURES,

            "tfidf_bucket_max_rows":
                TFIDF_BUCKET_MAX_ROWS,

            "max_deterministic_candidates":
                MAX_DETERMINISTIC_CANDIDATES

        }

}


summary_path = os.path.join(

    OUTPUT_DIR,

    "stage4_summary.json"

)


with open(

    summary_path,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        summary,

        f,

        indent=2

    )


# ============================================================
# 48. FINAL HARD CHECK
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "STAGE 4 FINAL SANITY CHECK"
)

print(
    "============================================================"
)


# ------------------------------------------------------------
# Verify exact final header again.
# ------------------------------------------------------------

with open(

    FINAL_CANDIDATE_OUTPUT,

    "rb"

) as f:

    final_header = f.readline()


expected_header = (

    b"source1_entity_id"
    b"\t"
    b"candidate_entity_ids\n"

)


if final_header != expected_header:

    raise RuntimeError(

        "FINAL candidate_pairs.tsv HEADER IS INVALID."

    )


# ------------------------------------------------------------
# Verify row count.
# ------------------------------------------------------------

final_rows = (

    sum(

        1

        for _

        in open(

            FINAL_CANDIDATE_OUTPUT,

            "r",

            encoding="utf-8"

        )

    )

    -

    1

)


if final_rows != test_s1_count:

    raise RuntimeError(

        f"FINAL candidate_pairs.tsv has "
        f"{final_rows:,} rows but Test S1 has "
        f"{test_s1_count:,} rows."

    )


# ------------------------------------------------------------
# Verify pandas sees exactly two columns.
# ------------------------------------------------------------

final_sample = pd.read_csv(

    FINAL_CANDIDATE_OUTPUT,

    sep="\t",

    dtype=str,

    keep_default_na=False,

    nrows=10

)


if list(final_sample.columns) != [

    "source1_entity_id",

    "candidate_entity_ids"

]:

    raise RuntimeError(

        "Pandas does not parse the final candidate "
        "file as the expected two-column TSV."

    )


print(
    "Header: PASS"
)

print(
    "Row count: PASS"
)

print(
    "Two-column parsing: PASS"
)

print(
    "Candidate ID validation: PASS"
)

print(
    "Final output directory:",
    OUTPUT_DIR
)


# ============================================================
# 49. FINAL MESSAGE
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "✅ STAGE 4 REVISED COMPLETE"
)

print(
    "============================================================"
)


print(
    "\nFinal candidate file:"
)

print(
    FINAL_CANDIDATE_OUTPUT
)


print(
    "\nTrain candidate file:"
)

print(
    TRAIN_CANDIDATE_OUTPUT
)


print(
    "\nTest candidate file:"
)

print(
    TEST_CANDIDATE_OUTPUT
)


print(
    "\nStrategy audit:"
)

print(

    os.path.join(

        OUTPUT_DIR,

        "train_candidate_strategy_audit.tsv"

    )

)


print(
    "\nRecall report:"
)

print(
    recall_output_path
)


print(
    "\nNext stage:"
)

print(
    "STAGE 5 — PAIR-LEVEL FEATURE ENGINEERING"
)


print(
    "============================================================"
)


# ============================================================
# 50. CLOSE
# ============================================================

con.close()

gc.collect()

print(
    "\nDuckDB connection closed."
)
