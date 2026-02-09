#!/usr/bin/env python3
"""
Generate synthetic market data for Phase 4 validation

This script creates synthetic market data that mimics the structure of real
market data required by the stat arb system. Useful for testing code paths
and validating the Python 3 migration without access to real market data.

Usage:
    python3 scripts/generate_synthetic_data.py --start=20130101 --end=20130630 --stocks=200
    python3 scripts/generate_synthetic_data.py --start=20130101 --end=20130630 --stocks=500 --output=test_data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import argparse


def generate_synthetic_universe(output_dir, start_date, end_date, num_stocks=200):
    """
    Generate synthetic universe files

    Creates daily universe files with stock metadata including sid, ticker,
    status, country, currency, price, advp, and market cap.
    """
    dates = pd.bdate_range(start_date, end_date)

    print(f"Generating {len(dates)} universe files for {num_stocks} stocks...")

    for idx, date in enumerate(dates):
        if idx % 50 == 0:
            print(f"  Universe: {date.strftime('%Y-%m-%d')} ({idx+1}/{len(dates)})")

        sids = list(range(1, num_stocks + 1))
        tickers = [f"SYN{i:04d}" for i in sids]

        # Stable prices with small daily variation
        base_prices = np.random.uniform(10, 200, num_stocks)
        daily_noise = np.random.normal(0, 0.01, num_stocks)
        prices = base_prices * (1 + daily_noise)

        df = pd.DataFrame({
            'sid': sids,
            'ticker_root': tickers,
            'status': 'ACTIVE',
            'country': 'USA',
            'currency': 'USD',
            'price': prices,
            'advp': np.random.uniform(1e6, 1e9, num_stocks),
            'mkt_cap': prices * np.random.uniform(1e8, 1e10, num_stocks)
        })

        # Save to universe/YYYY/YYYYMMDD.csv
        year_dir = os.path.join(output_dir, 'universe', str(date.year))
        os.makedirs(year_dir, exist_ok=True)
        df.to_csv(os.path.join(year_dir, f"{date.strftime('%Y%m%d')}.csv"), index=False)

    print(f"✅ Generated {len(dates)} universe files")


def generate_synthetic_prices(output_dir, start_date, end_date, num_stocks=200):
    """
    Generate synthetic OHLCV data with realistic random walk

    Creates daily price files with open, high, low, close, volume, market cap,
    and average dollar volume. Prices follow a random walk with drift.
    """
    dates = pd.bdate_range(start_date, end_date)

    print(f"Generating {len(dates)} price files for {num_stocks} stocks...")

    # Initialize prices with random walk
    prices = np.random.uniform(50, 150, num_stocks)

    for idx, date in enumerate(dates):
        if idx % 50 == 0:
            print(f"  Prices: {date.strftime('%Y-%m-%d')} ({idx+1}/{len(dates)})")

        # Random walk with slight positive drift
        returns = np.random.normal(0.0005, 0.02, num_stocks)
        prices = prices * (1 + returns)

        sids = list(range(1, num_stocks + 1))
        tickers = [f"SYN{i:04d}" for i in sids]

        # Generate OHLC around close with proper relationships
        closes = prices.copy()
        opens = closes * np.random.uniform(0.98, 1.02, num_stocks)

        # Ensure high >= max(open, close) and low <= min(open, close)
        max_oc = np.maximum(opens, closes)
        min_oc = np.minimum(opens, closes)
        highs = max_oc * np.random.uniform(1.0, 1.03, num_stocks)
        lows = min_oc * np.random.uniform(0.97, 1.0, num_stocks)

        volumes = np.random.uniform(1e6, 1e8, num_stocks).astype(int)

        df = pd.DataFrame({
            'sid': sids,
            'ticker': tickers,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'mkt_cap': closes * volumes * np.random.uniform(10, 100, num_stocks),
            'advp': closes * volumes
        })

        year_dir = os.path.join(output_dir, 'prices', str(date.year))
        os.makedirs(year_dir, exist_ok=True)
        df.to_csv(os.path.join(year_dir, f"{date.strftime('%Y%m%d')}.csv"), index=False)

    print(f"✅ Generated {len(dates)} price files")


def generate_synthetic_barra(output_dir, start_date, end_date, num_stocks=200):
    """
    Generate synthetic Barra factor exposures

    Creates daily Barra files with 13 standard factors, 58 industry dummies,
    residual returns, specific risk, and estimation universe flags.
    """
    dates = pd.bdate_range(start_date, end_date)

    print(f"Generating {len(dates)} Barra files for {num_stocks} stocks...")

    # Assign each stock to a random industry (1-58)
    stock_industries = np.random.randint(1, 59, num_stocks)

    for idx, date in enumerate(dates):
        if idx % 50 == 0:
            print(f"  Barra: {date.strftime('%Y-%m-%d')} ({idx+1}/{len(dates)})")

        sids = list(range(1, num_stocks + 1))

        # 13 Barra factors (standardized to mean=0, std=1)
        factors = pd.DataFrame({
            'sid': sids,
            'country': 1.0,  # Constant country factor
            'growth': np.random.normal(0, 1, num_stocks),
            'size': np.random.normal(0, 1, num_stocks),
            'sizenl': np.random.normal(0, 1, num_stocks),
            'divyild': np.random.normal(0, 1, num_stocks),
            'btop': np.random.normal(0, 1, num_stocks),
            'earnyild': np.random.normal(0, 1, num_stocks),
            'beta': np.random.normal(1.0, 0.3, num_stocks),
            'resvol': np.random.uniform(0.1, 0.5, num_stocks),
            'betanl': np.random.normal(0, 1, num_stocks),
            'momentum': np.random.normal(0, 1, num_stocks),
            'leverage': np.random.normal(0, 1, num_stocks),
            'liquidty': np.random.normal(0, 1, num_stocks),
            'barraResidRet': np.random.normal(0, 0.01, num_stocks),
            'barraSpecRisk': np.random.uniform(0.01, 0.05, num_stocks),
            'estu_barra4s': 1
        })

        # Add 58 industry dummies (each stock in one industry)
        for i in range(1, 59):
            factors[f'ind{i}'] = (stock_industries == i).astype(float)

        year_dir = os.path.join(output_dir, 'barra', str(date.year))
        os.makedirs(year_dir, exist_ok=True)
        factors.to_csv(os.path.join(year_dir, f"{date.strftime('%Y%m%d')}.csv"), index=False)

    print(f"✅ Generated {len(dates)} Barra files")


def generate_summary_stats(output_dir, start_date, end_date, num_stocks):
    """Generate summary statistics about the synthetic data"""
    dates = pd.bdate_range(start_date, end_date)

    summary = f"""
# Synthetic Data Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Output Directory:** {output_dir}

## Configuration

- **Start Date:** {start_date}
- **End Date:** {end_date}
- **Trading Days:** {len(dates)}
- **Number of Stocks:** {num_stocks}

## Files Generated

### Universe Files
- **Location:** {output_dir}/universe/YYYY/YYYYMMDD.csv
- **Files:** {len(dates)}
- **Columns:** sid, ticker_root, status, country, currency, price, advp, mkt_cap
- **Stocks per file:** {num_stocks}

### Price Files
- **Location:** {output_dir}/prices/YYYY/YYYYMMDD.csv
- **Files:** {len(dates)}
- **Columns:** sid, ticker, open, high, low, close, volume, mkt_cap, advp
- **Stocks per file:** {num_stocks}
- **Price Generation:** Random walk with drift (μ=0.0005, σ=0.02)
- **OHLC Validation:** high >= max(open, close), low <= min(open, close) ✅

### Barra Files
- **Location:** {output_dir}/barra/YYYY/YYYYMMDD.csv
- **Files:** {len(dates)}
- **Factors:** 13 (country, growth, size, sizenl, divyild, btop, earnyild, beta, resvol, betanl, momentum, leverage, liquidty)
- **Industries:** 58 (ind1 through ind58)
- **Additional:** barraResidRet, barraSpecRisk, estu_barra4s
- **Stocks per file:** {num_stocks}

## Total Storage

- **Estimated Size (Uncompressed):** {len(dates) * num_stocks * 0.001:.1f} MB
- **Estimated Size (Compressed):** {len(dates) * num_stocks * 0.0002:.1f} MB

## Configuration for loaddata.py

Update the following paths in `loaddata.py`:

```python
UNIV_BASE_DIR = "{output_dir}/universe"
PRICE_BASE_DIR = "{output_dir}/prices"
BARRA_BASE_DIR = "{output_dir}/barra"
```

## Limitations

**⚠️ SYNTHETIC DATA LIMITATIONS:**

1. **No Real Market Dynamics:** Prices follow random walk, no correlation structure
2. **No Corporate Actions:** No splits, dividends, or delistings
3. **No Intraday Data:** Bar files not generated (not needed for basic BSIM)
4. **No Fundamental Data:** Earnings, estimates, locates not generated
5. **Simplified Factors:** Barra factors are random, not calculated from fundamentals

**Use Case:** Code path testing and syntax validation only
**Validation Quality:** 30-40% (not suitable for numerical validation)

## Usage

Run a backtest with synthetic data:

```bash
# Update loaddata.py paths first
python3 bsim.py --start={start_date.replace('-', '')} --end={end_date.replace('-', '')} \\
    --fcast=hl:1:1 --kappa=2e-8 --maxnot=200e6
```

## Next Steps

1. ✅ Verify data was generated correctly
2. Update `loaddata.py` with paths above
3. Run import test: `python3 scripts/test_imports_py3.py`
4. Run basic backtest (see Usage above)
5. Check for crashes or errors
6. **For production validation:** Replace with real market data

---

**Note:** This is synthetic data for testing purposes only. For Phase 4 numerical
validation, you must acquire real market data (see docs/PHASE4_DATA_REQUIREMENTS.md).
"""

    with open(os.path.join(output_dir, 'SYNTHETIC_DATA_README.md'), 'w') as f:
        f.write(summary)

    print(f"\n✅ Summary written to {output_dir}/SYNTHETIC_DATA_README.md")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic market data for testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 6 months of data for 200 stocks
  python3 scripts/generate_synthetic_data.py --start=20130101 --end=20130630 --stocks=200

  # Generate 1 year of data for 500 stocks
  python3 scripts/generate_synthetic_data.py --start=20130101 --end=20131231 --stocks=500

  # Custom output directory
  python3 scripts/generate_synthetic_data.py --start=20130101 --end=20130630 --output=test_data
        """
    )

    parser.add_argument('--start', required=True, help='Start date (YYYYMMDD)')
    parser.add_argument('--end', required=True, help='End date (YYYYMMDD)')
    parser.add_argument('--stocks', type=int, default=200, help='Number of stocks (default: 200)')
    parser.add_argument('--output', default='synthetic_data', help='Output directory (default: synthetic_data)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Parse dates
    try:
        start_date = pd.to_datetime(args.start, format='%Y%m%d')
        end_date = pd.to_datetime(args.end, format='%Y%m%d')
    except ValueError as e:
        print(f"❌ Error parsing dates: {e}")
        print("   Use format YYYYMMDD (e.g., 20130101)")
        return 1

    if start_date >= end_date:
        print(f"❌ Error: Start date must be before end date")
        return 1

    print("="*70)
    print("SYNTHETIC DATA GENERATOR")
    print("="*70)
    print(f"Start Date:      {start_date.strftime('%Y-%m-%d')}")
    print(f"End Date:        {end_date.strftime('%Y-%m-%d')}")
    print(f"Trading Days:    {len(pd.bdate_range(start_date, end_date))}")
    print(f"Number of Stocks: {args.stocks}")
    print(f"Output Directory: {args.output}")
    print(f"Random Seed:     {args.seed}")
    print("="*70)
    print()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Generate data
    generate_synthetic_universe(args.output, start_date, end_date, args.stocks)
    generate_synthetic_prices(args.output, start_date, end_date, args.stocks)
    generate_synthetic_barra(args.output, start_date, end_date, args.stocks)
    generate_summary_stats(args.output, start_date, end_date, args.stocks)

    print()
    print("="*70)
    print("✅ SYNTHETIC DATA GENERATION COMPLETE")
    print("="*70)
    print()
    print(f"Next steps:")
    print(f"1. Review summary: {args.output}/SYNTHETIC_DATA_README.md")
    print(f"2. Update loaddata.py with paths:")
    print(f"   UNIV_BASE_DIR = '{args.output}/universe'")
    print(f"   PRICE_BASE_DIR = '{args.output}/prices'")
    print(f"   BARRA_BASE_DIR = '{args.output}/barra'")
    print(f"3. Run backtest: python3 bsim.py --start={args.start} --end={args.end} --fcast=hl:1:1")
    print()

    return 0


if __name__ == '__main__':
    exit(main())
