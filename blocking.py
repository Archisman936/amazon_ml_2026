# ============================================================
# AMAZON ML CHALLENGE 2026
# STAGE 4 — BLOCKING / CANDIDATE GENERATION
#
# INPUT:
#   Stage 3 normalized TSV files
#
# IMPORTANT:
#   INPUT_DIR = ""
#   means the files are uploaded directly into the current
#   Colab working directory (/content).
#
# OUTPUT:
#   /content/stage4_blocking/
#
# FINAL REQUIRED FILES:
#   train_candidate_pairs.tsv
#   test_candidate_pairs.tsv
#
# OPTIONAL AUDIT FILES:
#   train_candidate_strategy_audit.tsv
#   test_candidate_strategy_audit.tsv
#   blocking_report.tsv
#
# BLOCKING STRATEGIES:
#
#   1. Exact normalized-name block
#   2. Name + country
#   3. Name + postal/PIN
#   4. House-number + locality/city
#   5. Phonetic-name block
#   6. Transliteration-name block
#   7. Character TF-IDF n-gram Top-K retrieval
#
# DESIGN:
#
#   S1
#    |
#    +--> Block 1
#    +--> Block 2
#    +--> Block 3
#    +--> Block 4
#    +--> Block 5
#    +--> Block 6
#    +--> Block 7
#             |
#             v
#      UNION + DEDUPLICATION
#             |
#             v
#      candidate_pairs.tsv
#
# ============================================================


# ============================================================
# 1. INSTALL DEPENDENCIES
# ============================================================

%pip install -q duckdb scikit-learn numpy pandas tqdm


# ============================================================
# 2. IMPORTS
# ============================================================

import os
import re
import gc
import time
import shutil
import sqlite3
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 3. CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# IMPORTANT:
# Leave this as "" when the TSV files are uploaded directly
# into the Colab working directory.
#
# Example:
# INPUT_DIR = ""
#
# If your files are inside /content/mydata:
# INPUT_DIR = "/content/mydata"
# ------------------------------------------------------------

INPUT_DIR = ""


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUTPUT_DIR = "/content/stage4_blocking"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# Processing configuration
# ------------------------------------------------------------

CHUNK_SIZE = 100_000


# ------------------------------------------------------------
# Deterministic blocking
# ------------------------------------------------------------
#
# Very large blocks can create millions of unnecessary
# candidate pairs.
#
# A key whose bucket exceeds this value is skipped for that
# blocking strategy.
#
# Other blocking strategies and TF-IDF can still recover
# candidates.
# ------------------------------------------------------------

MAX_BLOCK_SIZE = 1_000


# ------------------------------------------------------------
# TF-IDF retrieval
# ------------------------------------------------------------

RUN_TFIDF = True

TFIDF_TOP_K = 15

TFIDF_MIN_COSINE = 0.25

TFIDF_TRIGGER_MAX_CANDIDATES = 30

TFIDF_NGRAM_RANGE = (3, 5)

TFIDF_MAX_FEATURES = 50_000

TFIDF_PREFIX_LENGTH = 2

TFIDF_MAX_IDS_PER_NAME = 20


# ------------------------------------------------------------
# Candidate strategy names
# ------------------------------------------------------------

STRATEGIES = [
    "exact_normalized_name",
    "name_country",
    "name_postal",
    "house_locality_city",
    "phonetic_name",
    "transliteration_name",
    "tfidf_char_ngram"
]


print("=" * 100)
print("AMAZON ML CHALLENGE 2026 — STAGE 4 BLOCKING")
print("=" * 100)

print("\nInput directory:")
print(
    os.path.abspath(
        INPUT_DIR
    )
)

print("\nOutput directory:")
print(
    OUTPUT_DIR
)

print("\nChunk size:")
print(
    f"{CHUNK_SIZE:,}"
)


# ============================================================
# 4. FILE DEFINITIONS
# ============================================================

FILES = {

    "train_source1":
        "train_source1_stage3_normalized.tsv",

    "train_source2":
        "train_source2_stage3_normalized.tsv",

    "train_source3":
        "train_source3_stage3_normalized.tsv",

    "test_source1":
        "test_source1_stage3_normalized.tsv",

    "test_source2":
        "test_source2_stage3_normalized.tsv",

    "test_source3":
        "test_source3_stage3_normalized.tsv",

    "train_ground_truth":
        "train_ground_truth.tsv",
}


# ============================================================
# 5. CHECK INPUT FILES
# ============================================================

print("\n" + "=" * 100)
print("INPUT FILE CHECK")
print("=" * 100)

AVAILABLE = {}

for name, filename in FILES.items():

    path = os.path.join(
        INPUT_DIR,
        filename
    )

    if os.path.exists(path):

        AVAILABLE[name] = path

        size_gb = (
            os.path.getsize(path)
            / (1024 ** 3)
        )

        print(
            f"✓ {name:<22}"
            f"{size_gb:>8.2f} GB"
        )

    else:

        print(
            f"✗ {name:<22}"
            f"NOT FOUND"
        )


# ============================================================
# 6. REQUIRED STAGE-3 COLUMNS
# ============================================================

REQUIRED_COLUMNS = [

    "entity_id",
    "business_name_normalized_original",
    "business_name_normalized_roman",
    "business_name_roman_phonetic",

    "address_house_number",
    "address_postal_code",
    "address_locality",
    "address_city",

    "country",
]


# ============================================================
# 7. DUCKDB DATABASE
# ============================================================
#
# DuckDB is used because your datasets contain millions of
# records and should not be loaded into pandas simultaneously.
#
# The database is stored on local Colab disk.
# ============================================================

DB_PATH = (
    "/content/"
    "amazon_ml_2026_stage4.duckdb"
)

if os.path.exists(DB_PATH):

    os.remove(DB_PATH)


con = duckdb.connect(
    DB_PATH
)

print(
    "\n✓ DuckDB initialized:"
)

print(
    DB_PATH
)


# ============================================================
# 8. HELPER — ESCAPE SQL PATH
# ============================================================

def sql_path(path):

    return path.replace(
        "'",
        "''"
    )


# ============================================================
# 9. LOAD SOURCE INTO DUCKDB
# ============================================================

def load_source_table(
    table_name,
    input_path
):

    print(
        f"\nLoading {table_name}"
    )

    print(
        input_path
    )

    escaped = sql_path(
        input_path
    )

    # --------------------------------------------------------
    # Load only the columns required by blocking.
    # --------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table_name} AS

        SELECT

            entity_id,

            business_name_normalized_original,
            business_name_normalized_roman,
            business_name_roman_phonetic,

            address_house_number,
            address_postal_code,
            address_locality,
            address_city,

            country

        FROM read_csv(
            '{escaped}',
            delim='\\t',
            header=true,
            all_varchar=true,
            ignore_errors=false
        );
        """
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    columns = [
        row[0]
        for row in con.execute(
            f"""
            DESCRIBE {table_name}
            """
        ).fetchall()
    ]

    missing = [
        c
        for c in REQUIRED_COLUMNS
        if c not in columns
    ]

    if missing:

        raise ValueError(
            f"{table_name} is missing columns: "
            f"{missing}"
        )

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {table_name}
        """
    ).fetchone()[0]

    print(
        f"✓ {table_name}: "
        f"{count:,} rows"
    )


# ============================================================
# 10. LOAD AVAILABLE TRAIN / TEST SOURCES
# ============================================================

for dataset_name in [

    "train_source1",
    "train_source2",
    "train_source3",

    "test_source1",
    "test_source2",
    "test_source3"

]:

    if dataset_name in AVAILABLE:

        load_source_table(
            dataset_name,
            AVAILABLE[dataset_name]
        )


# ============================================================
# 11. CREATE BLOCKING FEATURE TABLES
# ============================================================
#
# All keys are derived ONLY from the provided data.
#
# NO:
#   - external APIs
#   - geocoding
#   - business databases
#   - internet lookup
#
# This follows the challenge restriction on external lookup.
# ============================================================

def create_feature_table(
    source_table
):

    feature_table = (
        f"{source_table}_features"
    )

    print(
        f"\nCreating blocking keys:"
        f" {feature_table}"
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE
        {feature_table}
        AS

        SELECT

            entity_id,

            country,

            -- ---------------------------------------------
            -- Original normalized name
            -- ---------------------------------------------

            lower(
                trim(
                    coalesce(
                        business_name_normalized_original,
                        ''
                    )
                )
            ) AS name_key,

            -- ---------------------------------------------
            -- Transliteration/Roman normalized name
            -- ---------------------------------------------

            lower(
                trim(
                    coalesce(
                        business_name_normalized_roman,
                        ''
                    )
                )
            ) AS roman_name_key,

            -- ---------------------------------------------
            -- Phonetic name
            -- ---------------------------------------------

            lower(
                trim(
                    coalesce(
                        business_name_roman_phonetic,
                        ''
                    )
                )
            ) AS phonetic_key,

            -- ---------------------------------------------
            -- Country
            -- ---------------------------------------------

            lower(
                trim(
                    coalesce(
                        country,
                        ''
                    )
                )
            ) AS country_key,

            -- ---------------------------------------------
            -- Postal/PIN
            -- ---------------------------------------------

            lower(
                trim(
                    coalesce(
                        address_postal_code,
                        ''
                    )
                )
            ) AS postal_key,

            -- ---------------------------------------------
            -- House number
            -- ---------------------------------------------

            lower(
                trim(
                    coalesce(
                        address_house_number,
                        ''
                    )
                )
            ) AS house_key,

            -- ---------------------------------------------
            -- Locality
            -- ---------------------------------------------

            lower(
                trim(
                    coalesce(
                        address_locality,
                        ''
                    )
                )
            ) AS locality_key,

            -- ---------------------------------------------
            -- City
            -- ---------------------------------------------

            lower(
                trim(
                    coalesce(
                        address_city,
                        ''
                    )
                )
            ) AS city_key,

            -- ---------------------------------------------
            -- House + locality/city
            -- ---------------------------------------------

            CASE

                WHEN
                    trim(
                        coalesce(
                            address_house_number,
                            ''
                        )
                    ) <> ''

                    AND

                    (
                        trim(
                            coalesce(
                                address_locality,
                                ''
                            )
                        ) <> ''

                        OR

                        trim(
                            coalesce(
                                address_city,
                                ''
                            )
                        ) <> ''
                    )

                THEN

                    lower(
                        trim(
                            coalesce(
                                address_house_number,
                                ''
                            )
                        )
                    )
                    || '|' ||
                    lower(
                        trim(
                            coalesce(
                                nullif(
                                    address_locality,
                                    ''
                                ),
                                address_city,
                                ''
                            )
                        )
                    )

                ELSE ''

            END AS house_locality_key,

            -- ---------------------------------------------
            -- Name + country
            -- ---------------------------------------------

            CASE

                WHEN
                    trim(
                        coalesce(
                            business_name_normalized_original,
                            ''
                        )
                    ) <> ''

                    AND

                    trim(
                        coalesce(
                            country,
                            ''
                        )
                    ) <> ''

                THEN

                    lower(
                        trim(
                            business_name_normalized_original
                        )
                    )
                    || '|' ||
                    lower(
                        trim(
                            country
                        )
                    )

                ELSE ''

            END AS name_country_key,

            -- ---------------------------------------------
            -- Name + postal
            -- ---------------------------------------------

            CASE

                WHEN
                    trim(
                        coalesce(
                            business_name_normalized_original,
                            ''
                        )
                    ) <> ''

                    AND

                    trim(
                        coalesce(
                            address_postal_code,
                            ''
                        )
                    ) <> ''

                THEN

                    lower(
                        trim(
                            business_name_normalized_original
                        )
                    )
                    || '|' ||
                    lower(
                        trim(
                            address_postal_code
                        )
                    )

                ELSE ''

            END AS name_postal_key,

            -- ---------------------------------------------
            -- TF-IDF coarse prefix
            -- ---------------------------------------------
            #
            # First two alphanumeric Roman characters.
            #
            # Used only to make TF-IDF retrieval scalable.
            # It is NOT a final candidate strategy by itself.
            # ---------------------------------------------

            regexp_extract(
                lower(
                    coalesce(
                        business_name_normalized_roman,
                        ''
                    )
                ),
                '([a-z0-9]{2})',
                1
            ) AS tfidf_prefix

        FROM {source_table};
        """
    )

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {feature_table}
        """
    ).fetchone()[0]

    print(
        f"✓ {feature_table}: "
        f"{count:,} rows"
    )


for dataset_name in [

    "train_source1",
    "train_source2",
    "train_source3",

    "test_source1",
    "test_source2",
    "test_source3"

]:

    if dataset_name in AVAILABLE:

        create_feature_table(
            dataset_name
        )


# ============================================================
# 12. CREATE EMPTY CANDIDATE TABLES
# ============================================================

def create_candidate_table(
    prefix
):

    table_name = (
        f"{prefix}_candidate_pairs"
    )

    con.execute(
        f"""
        DROP TABLE IF EXISTS
        {table_name}
        """
    )

    con.execute(
        f"""
        CREATE TABLE
        {table_name} (

            source1_entity_id VARCHAR,

            candidate_entity_id VARCHAR,

            source_pair VARCHAR,

            strategy VARCHAR

        )
        """
    )

    return table_name


# ============================================================
# 13. ADD ONE BLOCKING STRATEGY
# ============================================================

def add_block_strategy(
    prefix,
    key_column,
    strategy_name,
    max_block_size=MAX_BLOCK_SIZE
):

    s1 = (
        f"{prefix}_source1_features"
    )

    s2 = (
        f"{prefix}_source2_features"
    )

    s3 = (
        f"{prefix}_source3_features"
    )

    candidate_table = (
        f"{prefix}_candidate_pairs"
    )

    print(
        "\n" + "-" * 100
    )

    print(
        f"BLOCK: {strategy_name}"
    )

    print(
        "-" * 100
    )

    # --------------------------------------------------------
    # S1 → S2
    # --------------------------------------------------------

    query_s2 = f"""

        WITH valid_keys AS (

            SELECT
                {key_column} AS block_key

            FROM {s2}

            WHERE
                {key_column} <> ''

            GROUP BY
                {key_column}

            HAVING
                COUNT(*) <= {max_block_size}
        )

        INSERT INTO {candidate_table}

        SELECT DISTINCT

            s1.entity_id
                AS source1_entity_id,

            s2.entity_id
                AS candidate_entity_id,

            'S1-S2'
                AS source_pair,

            '{strategy_name}'
                AS strategy

        FROM {s1} s1

        INNER JOIN valid_keys v
            ON s1.{key_column} = v.block_key

        INNER JOIN {s2} s2
            ON s2.{key_column} = v.block_key

        WHERE
            s1.{key_column} <> ''

    """

    con.execute(
        query_s2
    )


    # --------------------------------------------------------
    # S1 → S3
    # --------------------------------------------------------

    query_s3 = f"""

        WITH valid_keys AS (

            SELECT
                {key_column} AS block_key

            FROM {s3}

            WHERE
                {key_column} <> ''

            GROUP BY
                {key_column}

            HAVING
                COUNT(*) <= {max_block_size}
        )

        INSERT INTO {candidate_table}

        SELECT DISTINCT

            s1.entity_id
                AS source1_entity_id,

            s3.entity_id
                AS candidate_entity_id,

            'S1-S3'
                AS source_pair,

            '{strategy_name}'
                AS strategy

        FROM {s1} s1

        INNER JOIN valid_keys v
            ON s1.{key_column} = v.block_key

        INNER JOIN {s3} s3
            ON s3.{key_column} = v.block_key

        WHERE
            s1.{key_column} <> ''

    """

    con.execute(
        query_s3
    )


    # --------------------------------------------------------
    # Count
    # --------------------------------------------------------

    added = con.execute(
        f"""
        SELECT COUNT(*)

        FROM {candidate_table}

        WHERE strategy =
            '{strategy_name}'
        """
    ).fetchone()[0]


    print(
        f"✓ {strategy_name}: "
        f"{added:,} candidate rows"
    )


# ============================================================
# 14. RUN THE SIX DETERMINISTIC BLOCKS
# ============================================================

def run_deterministic_blocks(
    prefix
):

    create_candidate_table(
        prefix
    )

    # --------------------------------------------------------
    # 1. EXACT NORMALIZED NAME
    # --------------------------------------------------------

    add_block_strategy(
        prefix,
        "name_key",
        "exact_normalized_name"
    )


    # --------------------------------------------------------
    # 2. NAME + COUNTRY
    # --------------------------------------------------------

    add_block_strategy(
        prefix,
        "name_country_key",
        "name_country"
    )


    # --------------------------------------------------------
    # 3. NAME + POSTAL/PIN
    # --------------------------------------------------------

    add_block_strategy(
        prefix,
        "name_postal_key",
        "name_postal"
    )


    # --------------------------------------------------------
    # 4. HOUSE NUMBER + LOCALITY/CITY
    # --------------------------------------------------------

    add_block_strategy(
        prefix,
        "house_locality_key",
        "house_locality_city"
    )


    # --------------------------------------------------------
    # 5. PHONETIC NAME
    # --------------------------------------------------------

    add_block_strategy(
        prefix,
        "phonetic_key",
        "phonetic_name"
    )


    # --------------------------------------------------------
    # 6. TRANSLITERATION NAME
    # --------------------------------------------------------

    add_block_strategy(
        prefix,
        "roman_name_key",
        "transliteration_name"
    )


    print(
        "\n✓ Six deterministic blocks complete."
    )


# ============================================================
# 15. TF-IDF CANDIDATE COUNT
# ============================================================

def create_unresolved_table(
    prefix
):

    candidate_table = (
        f"{prefix}_candidate_pairs"
    )

    s1 = (
        f"{prefix}_source1_features"
    )

    unresolved = (
        f"{prefix}_tfidf_queries"
    )

    con.execute(
        f"""
        DROP TABLE IF EXISTS
        {unresolved}
        """
    )

    con.execute(
        f"""

        CREATE TABLE
        {unresolved}
        AS

        SELECT

            s1.entity_id
                AS source1_entity_id,

            s1.country_key
                AS country_key,

            s1.roman_name_key
                AS roman_name_key,

            s1.tfidf_prefix
                AS tfidf_prefix,

            COALESCE(
                COUNT(
                    DISTINCT cp.candidate_entity_id
                ),
                0
            ) AS candidate_count

        FROM {s1} s1

        LEFT JOIN {candidate_table} cp

            ON
                cp.source1_entity_id
                = s1.entity_id

        GROUP BY

            s1.entity_id,
            s1.country_key,
            s1.roman_name_key,
            s1.tfidf_prefix

        HAVING

            COUNT(
                DISTINCT cp.candidate_entity_id
            )
            < {TFIDF_TRIGGER_MAX_CANDIDATES}

        """
    )

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {unresolved}
        """
    ).fetchone()[0]

    print(
        f"TF-IDF queries required: "
        f"{count:,}"
    )


# ============================================================
# 16. LOAD TARGET TF-IDF BUCKET
# ============================================================

def get_tfidf_target_bucket(
    target_table,
    country_key,
    prefix
):

    query = f"""

        SELECT

            roman_name_key,

            list(
                entity_id
            ) AS entity_ids

        FROM {target_table}

        WHERE

            country_key = ?
            AND tfidf_prefix = ?
            AND roman_name_key <> ''

        GROUP BY

            roman_name_key

    """

    return con.execute(
        query,
        [
            country_key,
            prefix
        ]
    ).fetchall()


# ============================================================
# 17. GET TF-IDF QUERY BUCKET
# ============================================================

def get_tfidf_query_bucket(
    query_table,
    country_key,
    prefix
):

    query = f"""

        SELECT

            source1_entity_id,

            roman_name_key

        FROM {query_table}

        WHERE

            country_key = ?
            AND tfidf_prefix = ?

        ORDER BY
            source1_entity_id

    """

    return con.execute(
        query,
        [
            country_key,
            prefix
        ]
    ).fetchall()


# ============================================================
# 18. INSERT TF-IDF CANDIDATES
# ============================================================

def insert_tfidf_rows(
    candidate_table,
    rows
):

    if not rows:
        return 0

    df = pd.DataFrame(
        rows,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "source_pair",
            "strategy"
        ]
    )

    temp_name = (
        "tfidf_insert_temp"
    )

    con.register(
        temp_name,
        df
    )

    con.execute(
        f"""
        INSERT INTO {candidate_table}

        SELECT DISTINCT

            source1_entity_id,
            candidate_entity_id,
            source_pair,
            strategy

        FROM {temp_name}
        """
    )

    con.unregister(
        temp_name
    )

    return len(df)


# ============================================================
# 19. RUN TF-IDF FOR ONE TARGET SOURCE
# ============================================================

def run_tfidf_for_target(
    prefix,
    target_source_number
):

    candidate_table = (
        f"{prefix}_candidate_pairs"
    )

    query_table = (
        f"{prefix}_tfidf_queries"
    )

    target_table = (
        f"{prefix}_source"
        f"{target_source_number}_features"
    )

    print(
        "\n" + "=" * 100
    )

    print(
        f"TF-IDF RETRIEVAL → S{target_source_number}"
    )

    print(
        "=" * 100
    )


    # --------------------------------------------------------
    # Determine country/prefix buckets
    # --------------------------------------------------------

    buckets = con.execute(
        f"""
        SELECT DISTINCT

            country_key,
            tfidf_prefix

        FROM {query_table}

        WHERE

            tfidf_prefix <> ''
            AND country_key <> ''

        ORDER BY

            country_key,
            tfidf_prefix

        """
    ).fetchall()


    print(
        f"TF-IDF buckets: "
        f"{len(buckets):,}"
    )


    total_added = 0


    # --------------------------------------------------------
    # Bucket-by-bucket retrieval
    # --------------------------------------------------------

    for bucket_index, (
        country_key,
        prefix_key
    ) in enumerate(
        buckets,
        start=1
    ):

        # ----------------------------------------------------
        # Target unique names
        # ----------------------------------------------------

        target_rows = (
            get_tfidf_target_bucket(
                target_table,
                country_key,
                prefix_key
            )
        )


        if not target_rows:
            continue


        target_names = []

        target_ids = []


        for (
            name,
            ids
        ) in target_rows:

            if not name:
                continue


            ids = list(ids)


            # Avoid extremely large duplicate-name buckets
            if len(ids) > TFIDF_MAX_IDS_PER_NAME:

                ids = ids[
                    :TFIDF_MAX_IDS_PER_NAME
                ]


            target_names.append(
                name
            )

            target_ids.append(
                ids
            )


        if not target_names:
            continue


        # ----------------------------------------------------
        # S1 unresolved queries
        # ----------------------------------------------------

        query_rows = (
            get_tfidf_query_bucket(
                query_table,
                country_key,
                prefix_key
            )
        )


        if not query_rows:
            continue


        query_ids = [
            row[0]
            for row in query_rows
        ]


        query_names = [
            row[1]
            for row in query_rows
        ]


        # ----------------------------------------------------
        # TF-IDF model
        # ----------------------------------------------------

        vectorizer = TfidfVectorizer(

            analyzer="char",

            ngram_range=TFIDF_NGRAM_RANGE,

            lowercase=True,

            sublinear_tf=True,

            max_features=TFIDF_MAX_FEATURES,

            dtype=np.float32
        )


        try:

            X_target = (
                vectorizer.fit_transform(
                    target_names
                )
            )

        except ValueError:

            # Empty vocabulary
            continue


        if X_target.shape[1] == 0:
            continue


        # ----------------------------------------------------
        # Nearest-neighbour retrieval
        # ----------------------------------------------------

        n_neighbors = min(
            TFIDF_TOP_K,
            len(target_names)
        )


        nn = NearestNeighbors(

            n_neighbors=n_neighbors,

            metric="cosine",

            algorithm="brute",

            n_jobs=-1
        )


        nn.fit(
            X_target
        )


        X_query = (
            vectorizer.transform(
                query_names
            )
        )


        distances, indices = (
            nn.kneighbors(
                X_query
            )
        )


        # ----------------------------------------------------
        # Convert similarity into candidates
        # ----------------------------------------------------

        rows_to_insert = []


        for query_index in range(
            len(query_ids)
        ):

            source1_id = (
                query_ids[
                    query_index
                ]
            )


            for rank in range(
                n_neighbors
            ):

                distance = float(
                    distances[
                        query_index,
                        rank
                    ]
                )


                similarity = (
                    1.0 - distance
                )


                if similarity < (
                    TFIDF_MIN_COSINE
                ):

                    continue


                target_index = (
                    indices[
                        query_index,
                        rank
                    ]
                )


                candidate_ids = (
                    target_ids[
                        target_index
                    ]
                )


                for candidate_id in (
                    candidate_ids
                ):

                    rows_to_insert.append(

                        (

                            source1_id,

                            candidate_id,

                            f"S1-S{target_source_number}",

                            "tfidf_char_ngram"

                        )

                    )


        if rows_to_insert:

            total_added += (
                insert_tfidf_rows(
                    candidate_table,
                    rows_to_insert
                )
            )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            bucket_index == 1
            or bucket_index % 25 == 0
            or bucket_index == len(buckets)
        ):

            print(
                f"TF-IDF bucket "
                f"{bucket_index:,}/"
                f"{len(buckets):,}"
                f" | candidates added "
                f"{total_added:,}"
            )


        # ----------------------------------------------------
        # Free bucket memory
        # ----------------------------------------------------

        del X_target
        del X_query
        del nn
        del vectorizer
        del distances
        del indices

        gc.collect()


    print(
        f"\n✓ TF-IDF → S{target_source_number}"
        f" added {total_added:,} rows"
    )


# ============================================================
# 20. RUN TF-IDF FOR PREFIX
# ============================================================

def run_tfidf(
    prefix
):

    if not RUN_TFIDF:

        print(
            "\nTF-IDF disabled."
        )

        return


    create_unresolved_table(
        prefix
    )


    run_tfidf_for_target(
        prefix,
        2
    )


    run_tfidf_for_target(
        prefix,
        3
    )


# ============================================================
# 21. FINAL CANDIDATE TABLE
# ============================================================

def create_final_candidates(
    prefix
):

    raw_table = (
        f"{prefix}_candidate_pairs"
    )

    final_table = (
        f"{prefix}_candidate_final"
    )

    audit_table = (
        f"{prefix}_candidate_audit"
    )


    # --------------------------------------------------------
    # Remove duplicate S1 → candidate pairs
    # --------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE TABLE
        {final_table}
        AS

        SELECT

            source1_entity_id,

            candidate_entity_id,

            source_pair

        FROM {raw_table}

        GROUP BY

            source1_entity_id,
            candidate_entity_id,
            source_pair

        """
    )


    # --------------------------------------------------------
    # Strategy audit
    # --------------------------------------------------------
    #
    # One row per S1/candidate/source pair.
    # This keeps the exact strategy provenance.
    #
    # This is an internal audit artifact.
    # --------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE TABLE
        {audit_table}
        AS

        SELECT

            source1_entity_id,

            candidate_entity_id,

            source_pair,

            string_agg(
                DISTINCT strategy,
                '|' ORDER BY strategy
            ) AS strategies

        FROM {raw_table}

        GROUP BY

            source1_entity_id,
            candidate_entity_id,
            source_pair

        """
    )


    # --------------------------------------------------------
    # Required submission-format table
    # --------------------------------------------------------

    submission_table = (
        f"{prefix}_candidate_submission"
    )


    con.execute(
        f"""
        CREATE OR REPLACE TABLE
        {submission_table}
        AS

        SELECT

            s1.entity_id
                AS source1_entity_id,

            COALESCE(

                string_agg(

                    DISTINCT
                    c.candidate_entity_id,

                    ',' ORDER BY
                    c.candidate_entity_id

                ),

                ''

            ) AS candidate_entity_ids

        FROM
            {prefix}_source1_features s1

        LEFT JOIN
            {final_table} c

            ON
                c.source1_entity_id
                = s1.entity_id

        GROUP BY

            s1.entity_id

        ORDER BY

            s1.entity_id

        """
    )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_s1 = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {prefix}_source1_features
        """
    ).fetchone()[0]


    candidate_pairs = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {final_table}
        """
    ).fetchone()[0]


    avg_candidates = con.execute(
        f"""
        SELECT

            AVG(candidate_count)

        FROM (

            SELECT

                source1_entity_id,

                COUNT(*)
                    AS candidate_count

            FROM {final_table}

            GROUP BY
                source1_entity_id

        )

        """
    ).fetchone()[0]


    max_candidates = con.execute(
        f"""
        SELECT

            MAX(candidate_count)

        FROM (

            SELECT

                source1_entity_id,

                COUNT(*)
                    AS candidate_count

            FROM {final_table}

            GROUP BY
                source1_entity_id

        )

        """
    ).fetchone()[0]


    print(
        "\n" + "=" * 100
    )

    print(
        f"{prefix.upper()} CANDIDATE SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"S1 entities       : {total_s1:,}"
    )

    print(
        f"Unique candidates : {candidate_pairs:,}"
    )

    print(
        f"Average/S1        : "
        f"{avg_candidates or 0:.2f}"
    )

    print(
        f"Maximum/S1        : "
        f"{max_candidates or 0:,}"
    )


    return (
        final_table,
        audit_table,
        submission_table
    )


# ============================================================
# 22. EXPORT REQUIRED CANDIDATE_PAIRS.TSV
# ============================================================

def export_candidate_files(
    prefix
):

    (
        final_table,
        audit_table,
        submission_table
    ) = create_final_candidates(
        prefix
    )


    # --------------------------------------------------------
    # Required candidate_pairs.tsv
    # --------------------------------------------------------

    candidate_filename = (
        f"{prefix}_candidate_pairs.tsv"
    )

    candidate_path = os.path.join(
        OUTPUT_DIR,
        candidate_filename
    )


    candidate_path_sql = (
        sql_path(
            candidate_path
        )
    )


    con.execute(
        f"""
        COPY (

            SELECT

                source1_entity_id,
                candidate_entity_ids

            FROM {submission_table}

            ORDER BY
                source1_entity_id

        )

        TO '{candidate_path_sql}'

        WITH (

            HEADER true,

            DELIMITER '\\t'

        );

        """
    )


    # --------------------------------------------------------
    # Internal audit
    # --------------------------------------------------------

    audit_filename = (
        f"{prefix}_candidate_strategy_audit.tsv"
    )

    audit_path = os.path.join(
        OUTPUT_DIR,
        audit_filename
    )


    audit_path_sql = (
        sql_path(
            audit_path
        )
    )


    con.execute(
        f"""
        COPY (

            SELECT *

            FROM {audit_table}

            ORDER BY

                source1_entity_id,
                candidate_entity_id

        )

        TO '{audit_path_sql}'

        WITH (

            HEADER true,

            DELIMITER '\\t'

        );

        """
    )


    print(
        "\n✓ Required file:"
    )

    print(
        candidate_path
    )

    print(
        "\n✓ Internal strategy audit:"
    )

    print(
        audit_path
    )


    return candidate_path


# ============================================================
# 23. RUN COMPLETE BLOCKING PIPELINE
# ============================================================

def run_pipeline(
    prefix
):

    required = [

        f"{prefix}_source1",
        f"{prefix}_source2",
        f"{prefix}_source3"

    ]


    if not all(
        table in AVAILABLE
        for table in required
    ):

        print(
            f"\n⚠ Skipping {prefix}: "
            f"one or more source files are missing."
        )

        return None


    # --------------------------------------------------------
    # Deterministic strategies
    # --------------------------------------------------------

    start_time = time.time()


    run_deterministic_blocks(
        prefix
    )


    deterministic_time = (
        time.time()
        - start_time
    )


    print(
        f"\nDeterministic blocking time: "
        f"{deterministic_time / 60:.2f} minutes"
    )


    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    if RUN_TFIDF:

        tfidf_start = time.time()


        run_tfidf(
            prefix
        )


        tfidf_time = (
            time.time()
            - tfidf_start
        )


        print(
            f"\nTF-IDF time: "
            f"{tfidf_time / 60:.2f} minutes"
        )


    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    return export_candidate_files(
        prefix
    )


# ============================================================
# 24. TRAINING BLOCKING
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "TRAINING BLOCKING"
)

print(
    "=" * 100
)


train_candidate_path = (
    run_pipeline(
        "train"
    )
)


# ============================================================
# 25. TEST BLOCKING
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "TEST BLOCKING"
)

print(
    "=" * 100
)


test_candidate_path = (
    run_pipeline(
        "test"
    )
)


# ============================================================
# 26. TRAINING BLOCKING RECALL EVALUATION
# ============================================================
#
# Important:
# Training has ground truth.
#
# We measure:
#
#   blocking recall =
#   true matches found inside candidates
#   -----------------------------------
#   total true matches
#
# This tells us whether blocking is losing true matches
# BEFORE the matching model even gets a chance to see them.
#
# ============================================================

def evaluate_training_blocking():

    gt_path = AVAILABLE.get(
        "train_ground_truth"
    )


    if gt_path is None:

        print(
            "\n⚠ Training ground truth not found."
        )

        return


    gt_escaped = (
        sql_path(
            gt_path
        )
    )


    print(
        "\n" + "=" * 100
    )

    print(
        "TRAINING BLOCKING RECALL"
    )

    print(
        "=" * 100
    )


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    con.execute(
        f"""

        CREATE OR REPLACE TABLE
        train_ground_truth_eval
        AS

        SELECT

            source1_entity_id,

            matched_entity_ids

        FROM read_csv(

            '{gt_escaped}',

            delim='\\t',

            header=true,

            all_varchar=true

        )

        """
    )


    # --------------------------------------------------------
    # Convert comma-separated truth lists into one row
    # per true match.
    # --------------------------------------------------------

    con.execute(
        """

        CREATE OR REPLACE TABLE
        train_truth_pairs
        AS

        SELECT

            source1_entity_id,

            trim(
                unnest(
                    string_split(
                        coalesce(
                            matched_entity_ids,
                            ''
                        ),
                        ','
                    )
                )
            ) AS true_entity_id

        FROM train_ground_truth_eval

        WHERE

            coalesce(
                matched_entity_ids,
                ''
            ) <> ''

        """
    )


    # --------------------------------------------------------
    # Total true matches
    # --------------------------------------------------------

    total_true = con.execute(
        """

        SELECT COUNT(*)

        FROM train_truth_pairs

        """
    ).fetchone()[0]


    # --------------------------------------------------------
    # True matches recovered by UNION
    # --------------------------------------------------------

    total_recovered = con.execute(
        """

        SELECT COUNT(*)

        FROM train_truth_pairs t

        INNER JOIN train_candidate_pairs c

            ON
                c.source1_entity_id
                = t.source1_entity_id

                AND

                c.candidate_entity_id
                = t.true_entity_id

        """
    ).fetchone()[0]


    recall = (
        total_recovered / total_true
        if total_true
        else 0.0
    )


    print(
        f"Total true matches : "
        f"{total_true:,}"
    )

    print(
        f"Recovered          : "
        f"{total_recovered:,}"
    )

    print(
        f"Blocking recall    : "
        f"{recall:.6f}"
    )


    # --------------------------------------------------------
    # Per-strategy recall
    # --------------------------------------------------------

    strategy_results = []


    for strategy in STRATEGIES:

        recovered = con.execute(
            """

            SELECT COUNT(*)

            FROM (

                SELECT DISTINCT

                    source1_entity_id,
                    candidate_entity_id

                FROM train_candidate_pairs

                WHERE strategy = ?

            ) c

            INNER JOIN train_truth_pairs t

                ON
                    c.source1_entity_id
                    = t.source1_entity_id

                    AND

                    c.candidate_entity_id
                    = t.true_entity_id

            """,
            [
                strategy
            ]
        ).fetchone()[0]


        strategy_recall = (
            recovered / total_true
            if total_true
            else 0.0
        )


        candidate_count = con.execute(
            """

            SELECT COUNT(*)

            FROM train_candidate_pairs

            WHERE strategy = ?

            """,
            [
                strategy
            ]
        ).fetchone()[0]


        strategy_results.append(

            {

                "strategy":
                    strategy,

                "candidate_rows":
                    candidate_count,

                "true_matches_recovered":
                    recovered,

                "global_recall":
                    strategy_recall

            }

        )


    report = pd.DataFrame(
        strategy_results
    )


    report_path = os.path.join(
        OUTPUT_DIR,
        "blocking_strategy_report.tsv"
    )


    report.to_csv(
        report_path,
        sep="\t",
        index=False
    )


    print(
        "\nPer-strategy report:"
    )


    display(
        report
        .sort_values(
            "global_recall",
            ascending=False
        )
    )


    print(
        "\nSaved:"
    )

    print(
        report_path
    )


# ============================================================
# 27. RUN TRAINING RECALL CHECK
# ============================================================

if train_candidate_path is not None:

    evaluate_training_blocking()


# ============================================================
# 28. FINAL SANITY CHECK
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "FINAL STAGE 4 OUTPUT"
)

print(
    "=" * 100
)


for filename in sorted(
    os.listdir(
        OUTPUT_DIR
    )
):

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )


    if not os.path.isfile(path):

        continue


    size_mb = (
        os.path.getsize(path)
        / (1024 ** 2)
    )


    print(
        f"{filename:<60}"
        f"{size_mb:>10.2f} MB"
    )


# ============================================================
# 29. VERIFY REQUIRED FORMAT
# ============================================================

def verify_candidate_file(
    path
):

    if not os.path.exists(path):

        print(
            f"\n✗ Missing: {path}"
        )

        return


    sample = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        nrows=5
    )


    expected = [
        "source1_entity_id",
        "candidate_entity_ids"
    ]


    if list(sample.columns) != expected:

        print(
            f"\n✗ Wrong columns in {path}"
        )

        print(
            "Expected:",
            expected
        )

        print(
            "Actual:",
            list(sample.columns)
        )

        return


    print(
        f"\n✓ Valid candidate_pairs structure:"
    )

    print(
        path
    )

    print(
        "\nSample:"
    )

    display(
        sample
    )


if train_candidate_path is not None:

    verify_candidate_file(
        train_candidate_path
    )


if test_candidate_path is not None:

    verify_candidate_file(
        test_candidate_path
    )


# ============================================================
# 30. FINAL MESSAGE
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "✓ STAGE 4 — BLOCKING COMPLETE"
)

print(
    "=" * 100
)

print(
    "\nBlocking strategies used:"
)

for number, strategy in enumerate(
    STRATEGIES,
    start=1
):

    if (
        strategy
        == "tfidf_char_ngram"
        and not RUN_TFIDF
    ):

        print(
            f"{number}. {strategy} "
            f"[DISABLED]"
        )

    else:

        print(
            f"{number}. {strategy}"
        )


print(
    "\nCandidate generation:"
)

print(
    "Multiple blocks → UNION → DEDUPLICATION"
)


print(
    "\nFinal files:"
)

print(
    "candidate_pairs.tsv format:"
)

print(
    "source1_entity_id"
    "\\t"
    "candidate_entity_ids"
)


print(
    "\nNext stage:"
)

print(
    "candidate_pairs.tsv"
)

print(
    "      ↓"
)

print(
    "Pair feature engineering"
)

print(
    "      ↓"
)

print(
    "ML matching model"
)

print(
    "      ↓"
)

print(
    "matching_results.tsv"
)


# ============================================================
# 31. CLOSE DATABASE
# ============================================================

con.close()

print(
    "\n✓ DuckDB connection closed."
)

print(
    "✓ Stage 4 files remain in:"
)

print(
    OUTPUT_DIR
)
