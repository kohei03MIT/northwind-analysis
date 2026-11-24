"""Monthly revenue by customer country for the Northwind dataset.

Answers business question 8: which countries are growing and which are
flat or shrinking. Writes a tidy CSV, a summary CSV, and two charts.

Run from the repository root:
    .venv/bin/python analysis/country_growth.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Render to files only; no display needed.
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "Northwind Database.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"

# Countries with fewer orders than this over the whole period are pooled
# into one "Other" group, because a few large orders swing their monthly
# and year-over-year figures too much to rank them individually.
MIN_ORDERS = 20

# A year-over-year percentage is flagged as unreliable when the earlier
# window has fewer orders than this.
MIN_BASE_ORDERS = 5

# Same calendar months in consecutive years. 1996 and 1998 are partial
# years, so this is the only fair year-over-year comparison.
WINDOW_PRIOR = ("1997-01-01", "1997-05-01")  # Jan-Apr 1997
WINDOW_RECENT = ("1998-01-01", "1998-05-01")  # Jan-Apr 1998

# Chart colors (light-mode reference palette: one blue hue, two shades).
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
BLUE_LIGHT = "#b7d3f6"
BLUE_PRIOR = "#86b6ef"
BLUE_RECENT = "#1c5cab"


def load_order_revenue(path):
    """Build one row per order with its revenue and customer country.

    Revenue comes only from the Orders Details sheet; the hand-made
    price columns in the Product sheet are not used.

    Args:
        path (pathlib.Path): Northwind Excel workbook.

    Returns:
        pandas.DataFrame: Columns ``OrderID``, ``OrderDate``,
            ``CustomerID``, ``Country``, ``Revenue``, and ``Month``
            (``pandas.Period``, monthly).

    Raises:
        ValueError: If an order has no line items or its customer is
            missing from the Customers sheet.
    """
    sheets = pd.read_excel(
        path, sheet_name=["Orders", "Orders Details", "Customers"]
    )
    details = sheets["Orders Details"]
    orders = sheets["Orders"][["OrderID", "CustomerID", "OrderDate"]]
    customers = sheets["Customers"][["CustomerID", "Country"]]

    # Net line revenue after discount (Discount is a fraction, 0-0.25).
    details = details.assign(
        Revenue=details["UnitPrice"]
        * details["Quantity"]
        * (1 - details["Discount"])
    )
    order_revenue = details.groupby("OrderID", as_index=False)["Revenue"].sum()

    # validate= makes the script fail loudly if a key is ever duplicated.
    df = (
        order_revenue.merge(orders, on="OrderID", how="inner",
                            validate="one_to_one")
        .merge(customers, on="CustomerID", how="inner",
               validate="many_to_one")
    )
    if len(df) != len(orders):
        raise ValueError("Some orders lack line items or a known customer.")

    df["Month"] = df["OrderDate"].dt.to_period("M")
    return df


def find_complete_months(df):
    """Return the months whose order dates cover the whole month.

    The last month is complete only if an order falls within its final
    week. The first month is kept when data starts in its first week,
    which treats July 1996 (data starts on the 4th) as complete.

    Args:
        df (pandas.DataFrame): Order rows from ``load_order_revenue``.

    Returns:
        pandas.PeriodIndex: Months to use for trends, in order.
    """
    first_day = df["OrderDate"].min()
    last_day = df["OrderDate"].max()
    months = pd.period_range(first_day, last_day, freq="M")

    if first_day.day > 7:
        months = months[1:]
    if last_day.day < last_day.days_in_month - 7:
        months = months[:-1]
    return months


def assign_country_group(df):
    """Pool low-volume countries into a single "Other" group.

    Args:
        df (pandas.DataFrame): Order rows from ``load_order_revenue``.

    Returns:
        tuple[pandas.DataFrame, list[str]]: The rows with a new
            ``Group`` column, and the sorted names of the countries
            pooled into "Other".
    """
    orders_per_country = df.groupby("Country")["OrderID"].nunique()
    small = sorted(orders_per_country[orders_per_country < MIN_ORDERS].index)
    label = f"Other ({len(small)} countries)"
    df = df.assign(
        Group=df["Country"].where(~df["Country"].isin(small), label)
    )
    return df, small


def monthly_revenue_by_country(df, months):
    """Tabulate revenue for every country and month, including zeros.

    Args:
        df (pandas.DataFrame): Order rows with a ``Group`` column.
        months (pandas.PeriodIndex): Months to keep.

    Returns:
        pandas.DataFrame: Tidy table with columns ``Month``,
            ``Country``, ``Group``, ``Orders``, and ``Revenue``; months
            with no orders appear with zeros.
    """
    groups = df[["Country", "Group"]].drop_duplicates()
    grid = pd.MultiIndex.from_product(
        [months, sorted(df["Country"].unique())], names=["Month", "Country"]
    )
    table = (
        df[df["Month"].isin(months)]
        .groupby(["Month", "Country"])
        .agg(Orders=("OrderID", "nunique"), Revenue=("Revenue", "sum"))
        .reindex(grid, fill_value=0)
        .reset_index()
        .merge(groups, on="Country", how="left")
    )
    return table[["Month", "Country", "Group", "Orders", "Revenue"]]


def window_revenue(df, window, key):
    """Sum revenue and count orders inside a date window.

    Args:
        df (pandas.DataFrame): Order rows from ``load_order_revenue``.
        window (tuple[str, str]): Start date (inclusive) and end date
            (exclusive), ISO format.
        key (str): Column to group by, such as ``"Group"``.

    Returns:
        pandas.DataFrame: Indexed by ``key``, with ``Revenue`` and
            ``Orders`` columns.
    """
    start, end = window
    inside = df[(df["OrderDate"] >= start) & (df["OrderDate"] < end)]
    return inside.groupby(key).agg(
        Revenue=("Revenue", "sum"), Orders=("OrderID", "nunique")
    )


def summarize_groups(df):
    """Summarize all-time and Jan-Apr year-over-year revenue per group.

    Args:
        df (pandas.DataFrame): Order rows with a ``Group`` column.

    Returns:
        pandas.DataFrame: One row per group plus a "Total" row, with
            all-time ``Orders``, ``Customers``, ``Revenue``, and
            ``Share`` (fraction of all revenue), Jan-Apr revenue and
            orders for 1997 and 1998, and ``YoY`` (fractional change;
            NaN when Jan-Apr 1997 revenue is zero). Sorted by all-time
            revenue, with "Total" last.
    """
    def summarize(frame, key):
        """Aggregate all-time and Jan-Apr window figures per key.

        Args:
            frame (pandas.DataFrame): Order rows to aggregate.
            key (str): Column to group by.

        Returns:
            pandas.DataFrame: Indexed by ``key``, with ``Orders``,
                ``Customers``, ``Revenue``, and the Jan-Apr 1997 and
                1998 ``Revenue`` and ``Orders`` columns (0 if none).
        """
        overall = frame.groupby(key).agg(
            Orders=("OrderID", "nunique"),
            Customers=("CustomerID", "nunique"),
            Revenue=("Revenue", "sum"),
        )
        prior = window_revenue(frame, WINDOW_PRIOR, key)
        recent = window_revenue(frame, WINDOW_RECENT, key)
        out = overall.join(prior.add_suffix("JanApr1997"))
        out = out.join(recent.add_suffix("JanApr1998"))
        return out.fillna(0)

    by_group = summarize(df, "Group").sort_values("Revenue", ascending=False)
    total = summarize(df.assign(Group="Total"), "Group")
    out = pd.concat([by_group, total])
    out["Share"] = out["Revenue"] / total["Revenue"].iloc[0]
    out["YoY"] = (
        out["RevenueJanApr1998"] / out["RevenueJanApr1997"] - 1
    ).where(out["RevenueJanApr1997"] > 0)
    return out


def _money(value, _pos=None):
    """Format a dollar amount compactly for axis ticks.

    Args:
        value (float): Amount in dollars.
        _pos (int | None): Tick position; unused, but matplotlib
            passes it.

    Returns:
        str: Amount such as ``"$40k"`` or ``"$0"``.
    """
    return f"${value / 1000:,.0f}k" if value else "$0"


def _style_axes(ax):
    """Apply the shared recessive grid and axis styling.

    Args:
        ax (matplotlib.axes.Axes): Axes to style in place.

    Returns:
        None.
    """
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_small_multiples(monthly, groups, out_path):
    """Draw monthly revenue per country group as a grid of small panels.

    Each panel shows monthly revenue as light bars and the 3-month
    rolling mean as a line, on a shared y-axis so panels compare
    directly. Low-volume countries appear as one pooled "Other" panel.

    Args:
        monthly (pandas.DataFrame): Output of
            ``monthly_revenue_by_country``.
        groups (list[str]): Group names in panel order: countries by
            all-time revenue, largest first, with "Other" last.
        out_path (pathlib.Path): PNG file to write.

    Returns:
        None. Writes the figure to ``out_path``.
    """
    wide = monthly.pivot_table(
        index="Month", columns="Group", values="Revenue", aggfunc="sum"
    )
    panels = {g: wide[g] for g in groups}

    x = wide.index.to_timestamp()
    y_max = max(s.max() for s in panels.values()) * 1.08

    n_cols = 3
    n_rows = -(-len(panels) // n_cols)  # Ceiling division.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 2.55 * n_rows + 0.9),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)  # Unused slots in the last row.
    for ax, (name, series) in zip(axes.flat, panels.items()):
        _style_axes(ax)
        ax.bar(x, series.values, width=24, color=BLUE_LIGHT, linewidth=0)
        rolling = series.rolling(3, min_periods=3).mean()
        ax.plot(x, rolling.values, color=BLUE, linewidth=2)
        ax.set_title(name, loc="left", fontsize=10, color=INK, pad=4)
        ax.set_ylim(0, y_max)
        ax.yaxis.set_major_formatter(FuncFormatter(_money))
        ax.xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter("%b\n%Y")
        )
        ax.set_xticks(pd.to_datetime(["1996-07-01", "1997-07-01",
                                      "1998-04-01"]))

    # A panel directly above an empty slot is the bottom of its column,
    # so it needs its own date labels.
    for ax in axes[-2, len(panels) % n_cols:] if len(panels) % n_cols else []:
        ax.xaxis.set_tick_params(labelbottom=True)

    # Place the header in inches from the top so it holds for any height.
    height = fig.get_figheight()
    fig.suptitle(
        "Monthly revenue by customer country, Jul 1996 – Apr 1998",
        x=0.06, ha="left", fontsize=13, color=INK, y=1 - 0.12 / height,
    )
    fig.text(
        0.06, 1 - 0.5 / height,
        "Bars: monthly revenue after discounts. Line: 3-month rolling "
        "average. Same y-axis in every panel. May 1998 (6 days) excluded.\n"
        f"Other = countries with fewer than {MIN_ORDERS} orders in total.",
        fontsize=9, color=INK_SECONDARY, va="center",
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.45 / height,
                        top=1 - 1.05 / height, hspace=0.45, wspace=0.08)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_yoy_dumbbell(summary, out_path):
    """Draw Jan-Apr 1997 vs Jan-Apr 1998 revenue per group as dumbbells.

    Args:
        summary (pandas.DataFrame): Output of ``summarize_groups``.
        out_path (pathlib.Path): PNG file to write.

    Returns:
        None. Writes the figure to ``out_path``.
    """
    rows = summary.drop(index="Total").sort_values("RevenueJanApr1998")
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines["bottom"].set_visible(False)

    ax.hlines(y, rows["RevenueJanApr1997"], rows["RevenueJanApr1998"],
              color=BASELINE, linewidth=2)
    ax.scatter(rows["RevenueJanApr1997"], y, s=64, color=BLUE_PRIOR,
               zorder=3, label="Jan–Apr 1997")
    ax.scatter(rows["RevenueJanApr1998"], y, s=64, color=BLUE_RECENT,
               zorder=3, label="Jan–Apr 1998")

    # Label each row with its year-over-year change, right of both dots.
    # A star marks changes computed on a base of too few orders.
    for i, (_, r) in enumerate(rows.iterrows()):
        right = max(r["RevenueJanApr1997"], r["RevenueJanApr1998"])
        change = "n/a" if pd.isna(r["YoY"]) else f"{r['YoY']:+.0%}"
        low_base = r["OrdersJanApr1997"] < MIN_BASE_ORDERS
        ax.text(right + 2500, i, change + ("*" if low_base else ""),
                va="center", fontsize=8,
                color=INK_MUTED if low_base else INK_SECONDARY)

    ax.set_yticks(list(y))
    ax.set_yticklabels(rows.index, fontsize=9, color=INK)
    ax.xaxis.set_major_formatter(FuncFormatter(_money))
    ax.set_xlim(0, rows[["RevenueJanApr1997", "RevenueJanApr1998"]]
                .to_numpy().max() * 1.12)
    ax.legend(loc="lower right", frameon=False, fontsize=9,
              labelcolor=INK_SECONDARY)
    ax.set_title(
        "Revenue by customer country, Jan–Apr 1997 vs Jan–Apr 1998",
        loc="left", fontsize=12, color=INK, pad=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(
        0.02, 0.015,
        f"* Fewer than {MIN_BASE_ORDERS} orders in Jan–Apr 1997, so the "
        f"percentage is unreliable. Other = countries with fewer than "
        f"{MIN_ORDERS} orders in total.",
        fontsize=8, color=INK_MUTED,
    )
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    """Run the analysis and write all outputs.

    Args:
        None.

    Returns:
        None. Writes CSVs to ``data/processed/`` and PNGs to
        ``figures/``, and prints the key numbers.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    orders = load_order_revenue(DATA_FILE)
    months = find_complete_months(orders)
    orders, small = assign_country_group(orders)

    monthly = monthly_revenue_by_country(orders, months)
    monthly.assign(Month=monthly["Month"].astype(str)).to_csv(
        PROCESSED_DIR / "monthly_revenue_by_country.csv", index=False
    )

    summary = summarize_groups(orders)
    summary.to_csv(PROCESSED_DIR / "country_summary.csv",
                   index_label="Group")

    # Panel order: countries by revenue, then the pooled "Other" group.
    named = summary.drop(index="Total").index
    groups = [g for g in named if not g.startswith("Other")]
    groups += [g for g in named if g.startswith("Other")]
    plot_small_multiples(monthly, groups,
                         FIGURES_DIR / "monthly_revenue_by_country.png")
    plot_yoy_dumbbell(summary, FIGURES_DIR / "country_yoy_jan_apr.png")

    excluded = orders[~orders["Month"].isin(months)]
    print(f"Orders: {len(orders)}, revenue ${orders['Revenue'].sum():,.2f}")
    print(f"Trend months: {months[0]} to {months[-1]} ({len(months)}); "
          f"excluded {excluded['Month'].astype(str).unique().tolist()} "
          f"({len(excluded)} orders, ${excluded['Revenue'].sum():,.0f})")
    print(f"Pooled as Other (<{MIN_ORDERS} orders): {', '.join(small)}")
    cols = ["Orders", "Customers", "Revenue", "Share", "OrdersJanApr1997",
            "RevenueJanApr1997", "OrdersJanApr1998", "RevenueJanApr1998",
            "YoY"]
    print(summary[cols].to_string(formatters={
        "Revenue": "{:,.0f}".format,
        "RevenueJanApr1997": "{:,.0f}".format,
        "RevenueJanApr1998": "{:,.0f}".format,
        "Share": "{:.1%}".format,
        "YoY": lambda v: "n/a" if pd.isna(v) else f"{v:+.0%}",
    }))


if __name__ == "__main__":
    main()
