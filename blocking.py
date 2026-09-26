# ============================================================
# STAGE 4 — LIVE PROGRESS PATCH
# ============================================================
# Shows heartbeat output while long DuckDB blocking queries run.
# No row-count completeness checks are added.
# ============================================================

import time
import threading
import gc


# ============================================================
# 1. HEARTBEAT EXECUTOR
# ============================================================

def execute_with_heartbeat(
    con,
    sql,
    label,
    heartbeat_seconds=15
):
    """
    Execute a DuckDB query while printing elapsed-time progress.
    """

    finished = threading.Event()
    start_time = time.time()

    def heartbeat():

        while not finished.wait(heartbeat_seconds):

            elapsed = (
                time.time() - start_time
            )

            print(
                f"[RUNNING] {label} | "
                f"elapsed = {elapsed / 60:.1f} min",
                flush=True
            )

    thread = threading.Thread(
        target=heartbeat,
        daemon=True
    )

    thread.start()

    print(
        f"\n[START] {label}",
        flush=True
    )

    try:

        result = con.execute(sql)

        elapsed = (
            time.time() - start_time
        )

        print(
            f"[DONE] {label} | "
            f"time = {elapsed / 60:.2f} min",
            flush=True
        )

        return result

    finally:

        finished.set()
        thread.join(timeout=1)


# ============================================================
# 2. EXACT BLOCK — WITH PROGRESS
# ============================================================

def add_exact_block(
    s1_table,
    sx_table,
    target_table,
    source_pair,
    key_column,
    strategy
):

    label = (
        f"{strategy} "
        f"[{source_pair}]"
    )

    sql = f"""

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

    execute_with_heartbeat(
        con,
        sql,
        label
    )


    # Show how many rows now exist
    total_rows = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {target_table}
        """
    ).fetchone()[0]


    print(
        f"[PROGRESS] {target_table}: "
        f"{total_rows:,} candidate rows",
        flush=True
    )


# ============================================================
# 3. HOUSE + LOCALITY / CITY — OPTIMIZED + PROGRESS
# ============================================================
#
# This version avoids recalculating the same GROUP BY subqueries
# during the large join.
#
# We split locality and city joins instead of using one large OR.
# ============================================================

def add_house_location_block(
    s1_table,
    sx_table,
    target_table,
    source_pair
):

    print(
        f"\n========== HOUSE/LOCATION "
        f"[{source_pair}] ==========",
        flush=True
    )


    # --------------------------------------------------------
    # STEP 1 — Create eligible blocking keys for S1
    # --------------------------------------------------------

    print(
        "[1/6] Preparing S1 locality keys...",
        flush=True
    )

    s1_hl_table = (
        f"tmp_{source_pair}_s1_hl"
    )

    execute_with_heartbeat(

        con,

        f"""
        CREATE OR REPLACE TEMP TABLE
        {sql_ident(s1_hl_table)}
        AS

        SELECT
            entity_id,
            house_locality_key

        FROM {s1_table}

        WHERE
            house_locality_key <> ''

        GROUP BY
            entity_id,
            house_locality_key;


        """,

        f"{source_pair} | prepare S1 house+locality"
    )


    # --------------------------------------------------------
    # STEP 2 — Eligible source locality keys
    # --------------------------------------------------------

    print(
        "[2/6] Preparing source locality keys...",
        flush=True
    )

    sx_hl_table = (
        f"tmp_{source_pair}_sx_hl"
    )

    execute_with_heartbeat(

        con,

        f"""
        CREATE OR REPLACE TEMP TABLE
        {sql_ident(sx_hl_table)}
        AS

        SELECT
            entity_id,
            house_locality_key

        FROM {sx_table}

        WHERE
            house_locality_key <> ''

        GROUP BY
            entity_id,
            house_locality_key

        QUALIFY COUNT(*) OVER (
            PARTITION BY house_locality_key
        ) <= {MAX_BLOCK_SIZE};

        """,

        f"{source_pair} | prepare source house+locality"
    )


    # --------------------------------------------------------
    # STEP 3 — Locality join
    # --------------------------------------------------------

    print(
        "[3/6] Running house + locality join...",
        flush=True
    )

    execute_with_heartbeat(

        con,

        f"""
        INSERT INTO {target_table}

        SELECT

            a.entity_id,

            b.entity_id,

            '{source_pair}',

            'house_locality'

        FROM {sql_ident(s1_hl_table)} a

        INNER JOIN {sql_ident(sx_hl_table)} b

            ON
                a.house_locality_key
                =
                b.house_locality_key;

        """,

        f"{source_pair} | house_locality join"
    )


    # --------------------------------------------------------
    # STEP 4 — Prepare city keys
    # --------------------------------------------------------

    print(
        "[4/6] Preparing city keys...",
        flush=True
    )

    s1_hc_table = (
        f"tmp_{source_pair}_s1_hc"
    )

    sx_hc_table = (
        f"tmp_{source_pair}_sx_hc"
    )


    execute_with_heartbeat(

        con,

        f"""
        CREATE OR REPLACE TEMP TABLE
        {sql_ident(s1_hc_table)}
        AS

        SELECT
            entity_id,
            house_city_key

        FROM {s1_table}

        WHERE
            house_city_key <> ''

        GROUP BY
            entity_id,
            house_city_key;

        """,

        f"{source_pair} | prepare S1 house+city"
    )


    execute_with_heartbeat(

        con,

        f"""
        CREATE OR REPLACE TEMP TABLE
        {sql_ident(sx_hc_table)}
        AS

        SELECT
            entity_id,
            house_city_key

        FROM {sx_table}

        WHERE
            house_city_key <> ''

        GROUP BY
            entity_id,
            house_city_key

        QUALIFY COUNT(*) OVER (
            PARTITION BY house_city_key
        ) <= {MAX_BLOCK_SIZE};

        """,

        f"{source_pair} | prepare source house+city"
    )


    # --------------------------------------------------------
    # STEP 5 — City join
    # --------------------------------------------------------

    print(
        "[5/6] Running house + city join...",
        flush=True
    )

    execute_with_heartbeat(

        con,

        f"""
        INSERT INTO {target_table}

        SELECT

            a.entity_id,

            b.entity_id,

            '{source_pair}',

            'house_locality'

        FROM {sql_ident(s1_hc_table)} a

        INNER JOIN {sql_ident(sx_hc_table)} b

            ON
                a.house_city_key
                =
                b.house_city_key;

        """,

        f"{source_pair} | house_city join"
    )


    # --------------------------------------------------------
    # STEP 6 — Current candidate count
    # --------------------------------------------------------

    total_rows = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {target_table}
        """
    ).fetchone()[0]


    print(
        f"[6/6] {target_table}: "
        f"{total_rows:,} rows",
        flush=True
    )


    # Cleanup
    con.execute(
        f"""
        DROP TABLE IF EXISTS
        {sql_ident(s1_hl_table)}
        """
    )

    con.execute(
        f"""
        DROP TABLE IF EXISTS
        {sql_ident(sx_hl_table)}
        """
    )

    con.execute(
        f"""
        DROP TABLE IF EXISTS
        {sql_ident(s1_hc_table)}
        """
    )

    con.execute(
        f"""
        DROP TABLE IF EXISTS
        {sql_ident(sx_hc_table)}
        """
    )

    gc.collect()


print(
    "\n✓ Live-progress blocking functions loaded."
)

print(
    "The next blocking query will print a heartbeat every 15 seconds."
)
