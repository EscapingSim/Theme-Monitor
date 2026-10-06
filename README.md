# Theme Monitor

Nightly dashboard ranking stock themes and the leaders within each.

## One-time setup (about 15 minutes)
1. Create a free account at github.com.
2. Click **New repository**, name it `theme-monitor`, choose **Public**, and create it.
   (GitHub Pages on a free account requires a public repo. Only your ticker list and prices are visible.)
3. On the repo page, click **uploading an existing file** and drag in everything from this folder,
   including the `.github` folder. Commit.
   - If the `.github` folder doesn't upload (hidden folder), create it by hand: **Add file > Create new file**,
     name it `.github/workflows/nightly.yml`, and paste in the contents of that file.
4. **Settings > Pages**: Source = *Deploy from a branch*, Branch = `main`, folder = `/docs`. Save.
5. **Actions** tab: enable workflows if prompted, open *Nightly theme update*, click **Run workflow** once.
6. After about 2 minutes your dashboard is live at `https://<your-username>.github.io/theme-monitor/`.
   Bookmark it on your phone.

It then runs automatically every weekday at 5:30 PM Mountain.

## Changing things
- **Tickers/themes:** edit `data/tickers.csv` (theme, subtheme, ticker, company, exposure).
  Exposure must be `Pure-play`, `Major segment` or `Diversified`.
- **Leader score weights:** `WEIGHTS` at the top of `build.py`.
- **Run time:** the `cron` line in `.github/workflows/nightly.yml` (UTC).

Daily snapshots are saved to `data/history/` for future trend charts.

## Setups page (Minervini / Zanger)
`docs/setups.html` screens the S&P 1500 (pulled from Wikipedia nightly) plus your theme stocks.
- **Add stocks outside the S&P 1500:** create `data/extra_universe.csv` with columns `ticker,name,sector`.
- **Liquidity floor, leading-theme count:** settings at the top of `build.py` (`MIN_PRICE`, `MIN_DOLLAR_VOL`, `LEADING_THEMES`).
- The universe list is cached in `data/universe_cache.csv` so a Wikipedia hiccup doesn't break the run.
