# northwind-analysis

Python analysis of the Northwind sales workbook, answering three business
questions for a sales manager:

1. **Country growth (question 8):** which customer countries are growing and
   which are shrinking? Monthly revenue by country, plus a Jan–Apr 1997 vs
   Jan–Apr 1998 comparison.
2. **Customer concentration (question 2):** how much revenue depends on a few
   customers? Each customer's share of revenue, the top-10 share, and a
   Pareto chart.
3. **Discounts (question 4):** do discounts increase order size, or just give
   away revenue? Discounted vs full-price lines for the same product at the
   same list price, plus how discounts are spread across customers. The data
   has no costs, so this shows revenue given away, not profit.

Revenue is always `UnitPrice × Quantity × (1 − Discount)` from the
Orders Details sheet. The manually entered columns in the Product sheet are not
used.

**Live site:** _add the GitHub Pages link here_ (served from `index.html` at the
repository root, branch `main`).

## Repository layout

| Path | Contents |
|---|---|
| `data/Northwind Database.xlsx` | Source workbook (input) |
| `analysis/` | The three analysis scripts |
| `data/processed/` | CSV tables written by the scripts |
| `figures/` | PNG charts written by the scripts |
| `index.html` | Single-page report that uses the figures |
| `requirements.txt` | Pinned Python packages |

## Reproduce every number, figure, and table

Requires Python 3.9 or later. Run all commands from the repository root.

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

2. Install the pinned packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the scripts in this order. Each one reads the workbook directly, prints
   its key numbers to the terminal, and overwrites its outputs.

   ```bash
   python analysis/country_growth.py
   python analysis/customer_concentration.py
   python analysis/discount_effect.py
   ```

   The other two scripts import shared code (revenue loader, paths, chart
   colors) from `country_growth.py`, and `discount_effect.py` also uses the
   customer ranking from `customer_concentration.py`. Each script still runs
   on its own; the order just matches the report.

4. Check the outputs:

   | Script | Tables in `data/processed/` | Figures in `figures/` |
   |---|---|---|
   | `country_growth.py` | `monthly_revenue_by_country.csv`, `country_summary.csv` | `monthly_revenue_by_country.png`, `country_yoy_jan_apr.png` |
   | `customer_concentration.py` | `customer_revenue.csv`, `customer_concentration.csv` | `customer_pareto.png` |
   | `discount_effect.py` | `discount_effect.csv`, `discount_by_customer.csv` | `discount_effect.png` |

   The discount confidence intervals use a fixed random seed, so repeated
   runs give identical results. `git status` should show no changes after a
   rerun.

5. View the report by opening `index.html` in a browser. It loads the PNGs
   from `figures/`.

## Data handling

- May 1998 has only 6 days of data, so it is left out of trends. Growth
  compares Jan–Apr 1997 with Jan–Apr 1998.
- The 8 countries with fewer than 20 orders are pooled as "Other".
- Customer concentration is compared only between periods of equal length.
- Two customers (FISSA, PARIS) never ordered and are excluded.

The "Data notes" section of `index.html` has the full details.
