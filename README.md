# Fish Fund

Zebrafish pick the stocks. You place the trades.

A webcam looks down on one tank. The tank is split into zones along its length, and each zone has a printed ticker card. Fish swim into a zone, the countdown ends, and the busiest zone wins. You buy or sell that name yourself at a brokerage, then log it here. The ledger compares your account with a simulated S&P 500 buy of the same starting cash.

Nothing in this project talks to a broker, and nothing sends orders.

## Setup

Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the commands from this directory. Data files are written next to the scripts, not into the current directory, so a full path works too.

Hardware this expects:

- One USB webcam above the tank, pointed down.
- A light tank bottom and small, dark or striped fish.
- Zones along the length of the tank. The default round uses 3. Draw more if a buy round needs more tickers.

There is no camera on a typical dev machine. `calibrate.py` and `pick.py` exit with a clear message in that case.

## Calibrate

```bash
python calibrate.py
```

The preview opens. Drag a rectangle around each zone. After each drag, the terminal asks for the ticker (`AAPL`, `VOO`, `BRK.B`). Zones you already saved are loaded on the next run.

Keys, also shown in the window title and along the bottom of the preview:

| Key | Action |
| --- | --- |
| drag | Draw a zone, then type its ticker in the terminal |
| `u` | Undo the last zone |
| `c` | Clear every zone so you can redraw |
| `s` | Save zones into `config.json` |
| `q` or Esc | Quit. If you have unsaved edits, it asks before closing |

Saving only replaces `zones`. Countdown, blob sizes, and the other settings stay as they are.

Draw order matters for the other rounds:

- Hold: the first zone is HOLD, the second is SELL.
- Size: the first three zones are 10%, 25%, and 50%.

Draw those left to right if you want the cards to match the labels. Zones should not overlap. A fish in an overlap counts for the earlier zone. Recalibrate if the camera resolution changes. Zone positions are pixels in that frame.

`--config path` reads and updates a different file.

## Blob size

`min_blob_area` and `max_blob_area` in `config.json` are the pixel area of a detected fish. Defaults are 100 and 8000, a wide band for a small fish seen from above. A zebrafish on a 720p or 1080p top-down view is often a few hundred to a few thousand pixels. Specks are smaller. Hands, reflections, and the filter are larger.

Tune while watching a round. Counted fish get a colored ring. Blobs outside every zone get a grey ring and are not counted.

- Fish never get a ring: lower `min_blob_area` (try 60, then 30).
- Gravel, bubbles, or reflections count: raise `min_blob_area`, or lower `max_blob_area`.
- Moving the camera or changing resolution changes these numbers. Tune again after that.

`background_mode` is `mog2` by default. That learns the tank from motion, which suits fish that are swimming. Fish that hover can disappear into the background. For a locked-off camera, shoot an empty tank instead:

```bash
python pick.py --capture-background
```

That saves `background.png`, sets `background_mode` to `frame`, and leaves the other settings alone. Later rounds subtract that still. If the light changes, capture it again.

## Monthly round

One round, then you trade, then you log, then you chart.

```bash
python pick.py --round buy
```

`python pick.py --round buy --video tank.mp4` plays that file through the same detection and overlay instead of the webcam.

The preview shows each zone, its ticker, a live fish count, and the countdown (`countdown_seconds`, default 15). The whole round, including the overlay, is saved to `recordings/<timestamp>.mp4`. When time is up, the zone with the most fish wins. If zones are tied, the higher average count over the final 3 seconds wins. If that is still tied, one of the remaining zones is chosen at random. A banner reads `FISH PICKED: <TICKER>` for `banner_seconds` (default 5). The result is appended to `picks.csv`.

Place that trade yourself in your brokerage. Then record the fill:

```bash
python fund.py buy AAPL --amount 50
python chart.py
```

`chart.py` writes `charts/fish-fund-YYYY-MM-DD.png` (1080×1920) and prints the path. The poster is the fish account against the S&P 500 since you started.

Other rounds:

```bash
python pick.py --round hold --ticker AAPL
python pick.py --round size
```

Hold uses the first two zones, labeled HOLD and SELL. The banner names HOLD or SELL, and `picks.csv` stores the ticker in the subject column. Size uses the first three zones, labeled 10%, 25%, and 50%. You still choose the dollars and place the order. If the fish pick 25%, that means 25% of whatever cash you decided to risk.

A buy round needs at least 1 zone, hold needs 2, and size needs 3.

## The ledger

`fund.py` only writes down what you already did.

```bash
python fund.py init --cash 200
python fund.py buy AAPL --amount 50
python fund.py sell AAPL --shares 0.1
python fund.py status
```

`init` stores the cash and a simulated SPY buy of that same cash at the latest price (`shares = cash / price`). It will not overwrite an existing `portfolio.json`.

`buy` spends dollars from cash on fractional shares at the latest price. It rejects an amount bigger than your cash. `sell` rejects a share count bigger than the holding and adds the proceeds to cash. `status` prints each holding (shares, price, value), cash, total value, fish return versus starting cash, S&P return, and who is ahead: `FISH`, `SPY`, or `TIE`.

Prices are the latest daily close from Yahoo Finance, so evenings and weekends use the last session. Dates are stored as `YYYY-MM-DD` in `America/New_York`. If a price cannot be fetched, the command says so and leaves `portfolio.json` unchanged.

Use another file by putting `--portfolio` before or after the command:

```bash
python fund.py --portfolio /tmp/book.json init --cash 200
python fund.py status --portfolio /tmp/book.json
```

`chart.py` takes the same `--portfolio` flag, plus `--output` for the PNG:

```bash
python chart.py --portfolio /tmp/book.json --output /tmp/fish-fund.png
```

The chart rebuilds the fish account from the trade history: cash and shares as of each day, valued at that day's close. The S&P line is the benchmark share count times the SPY close. A trade logged on a weekend shows up on the next close, or on that calendar date if the latest close is already behind it.

`portfolio.json`, `picks.csv`, recordings, and chart PNGs are gitignored.

## Config

`config.json` ships with empty zones and these defaults:

| Field | Default | Meaning |
| --- | --- | --- |
| `camera_index` | `0` | USB camera index |
| `countdown_seconds` | `15` | Length of a round |
| `banner_seconds` | `5` | How long the winner stays on screen |
| `min_blob_area` | `100` | Smallest fish blob, in pixels |
| `max_blob_area` | `8000` | Largest fish blob, in pixels |
| `background_mode` | `mog2` | `mog2` or `frame` |
| `background_image` | empty | Still frame used when mode is `frame` |
| `zones` | `[]` | `{ticker, x, y, w, h}` in camera pixels |
