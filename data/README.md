# Data — LOBSTER samples

## Where to get the data

LOBSTER provides free academic sample files for AAPL, GOOG, MSFT, INTC, AMZN.
Download from: **https://lobsterdata.com/info/DataSamples.php**

Each sample is a ZIP file containing two CSVs:

- `{TICKER}_{DATE}_{START}_{END}_message_{N}.csv` — event log
- `{TICKER}_{DATE}_{START}_{END}_orderbook_{N}.csv` — book snapshots after each event

Where `N` is the number of book levels (1, 5, 10, 30, 50).

## How to set up

```bash
# 1. Create the data dir (already exists if you followed setup)
mkdir -p data/raw

# 2. Download a sample (manually from the LOBSTER website)
#    Pick AAPL with 5 levels for first experiments — small enough to iterate fast.

# 3. Extract into data/raw/
unzip _data_dwn_5_15__AAPL_2012-06-21_34200000_57600000_5.zip -d data/raw/

# 4. Verify
ls data/raw/
# Should show:
#   AAPL_2012-06-21_34200000_57600000_message_5.csv
#   AAPL_2012-06-21_34200000_57600000_orderbook_5.csv
```

## File format reference

### Message file columns (no header)

| Col | Name      | Description |
|-----|-----------|-------------|
| 1   | time      | Seconds after midnight |
| 2   | type      | Event type (see below) |
| 3   | order_id  | Unique order identifier |
| 4   | size      | Shares |
| 5   | price     | Price in $0.0001 (integer) |
| 6   | direction | +1 = buy, -1 = sell |

**Event types:**
- 1 = Submission of a new limit order
- 2 = Cancellation (partial)
- 3 = Deletion (total)
- 4 = Execution of a visible limit order
- 5 = Execution of a hidden limit order
- 6 = Cross trade (auction)
- 7 = Trading halt

### Orderbook file columns (no header)

For each level `i = 1 .. N`:
- ask_price_i
- ask_size_i
- bid_price_i
- bid_size_i

Prices stored as integers in $0.0001. The `load_orderbook` helper in
`src/data/lobster.py` automatically scales to dollars.

## What's gitignored

Raw and processed data files are in `.gitignore` — they're not committed to the
repo. Each developer downloads the LOBSTER samples locally.
