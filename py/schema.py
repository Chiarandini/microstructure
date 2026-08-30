"""The exported event schema, shared by everything that reads it.

Kept in one place so that a column added on the Rust side has exactly one
Python definition to update, rather than three scripts that drift apart.
"""

# ITCH prices are fixed-point with four implied decimals.
TICK = 10_000

# Continuous session, 09:30 to 16:00 ET, as nanoseconds since midnight.
SESSION_OPEN_NS = int(9.5 * 3600 * 1e9)
SESSION_CLOSE_NS = int(16 * 3600 * 1e9)

COLUMNS = [
    "ts_ns",
    "event",
    "side",
    "price",
    "shares",
    "old_price",
    "old_shares",
    "printable",
    "resting_price",
    "bid_px_before",
    "ask_px_before",
    "bid_sz_before",
    "ask_sz_before",
    "bid_px_after",
    "ask_px_after",
    "bid_sz_after",
    "ask_sz_after",
]

# Columns that are only populated for some event types stay float64 so the
# empty field reads back as NaN rather than forcing a sentinel value.
DTYPES = {
    "ts_ns": "int64",
    "event": "category",
    "side": "category",
    "price": "int64",
    "shares": "int64",
    "old_price": "float64",
    "old_shares": "float64",
    "printable": "float64",
    "resting_price": "float64",
    "bid_px_before": "float64",
    "ask_px_before": "float64",
    "bid_sz_before": "int64",
    "ask_sz_before": "int64",
    "bid_px_after": "float64",
    "ask_px_after": "float64",
    "bid_sz_after": "int64",
    "ask_sz_after": "int64",
}
