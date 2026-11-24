"""Customer revenue concentration for the Northwind dataset.

Answers business question 2: how much revenue depends on a few
customers. Writes a per-customer CSV, a concentration summary CSV, and
a Pareto chart.

Run from the repository root:
    .venv/bin/python analysis/customer_concentration.py
"""

import matplotlib

matplotlib.use("Agg")  # Render to files only; no display needed.
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

# Shared loader, paths, and colors, so revenue is defined the same way
# in every analysis.
from country_growth import (
    BASELINE, BLUE, BLUE_LIGHT, DATA_FILE, FIGURES_DIR, GRID, INK,
    INK_MUTED, INK_SECONDARY, PROCESSED_DIR, SURFACE, _style_axes,
    load_order_revenue,
)

TOP_N = 10           # The headline "top customers" group.
BARS_SHOWN = 15      # Customers listed by name in the left panel.
GRAY = "#c3c2b7"     # De-emphasis color for customers outside the top N.

# Periods for checking whether concentration is changing. Shorter
# periods look more concentrated because fewer customers order in them,
# so trends are judged only between periods of equal length: Jan-Apr
# 1997 vs Jan-Apr 1998 (May 1998 has only 6 days of data).
PERIODS = {
    "All orders (Jul 1996 – May 1998)": (None, None),
    "Full year 1997": ("1997-01-01", "1998-01-01"),
    "Jan–Apr 1997": ("1997-01-01", "1997-05-01"),
    "Jan–Apr 1998": ("1998-01-01", "1998-05-01"),
}


def customer_revenue(orders, customers):
    """Rank customers by revenue and add share and cumulative share.

    Args:
        orders (pandas.DataFrame): Order rows from
            ``country_growth.load_order_revenue``.
        customers (pandas.DataFrame): The Customers sheet.

    Returns:
        pandas.DataFrame: One row per customer with at least one order,
            sorted by revenue (largest first), with ``Rank``,
            ``CustomerID``, ``CompanyName``, ``Country``, ``Orders``,
            ``FirstOrder``, ``LastOrder``, ``Revenue``, ``Share``, and
            ``CumulativeShare`` (shares are fractions of total revenue).
    """
    table = (
        orders.groupby("CustomerID")
        .agg(Orders=("OrderID", "nunique"),
             FirstOrder=("OrderDate", "min"),
             LastOrder=("OrderDate", "max"),
             Revenue=("Revenue", "sum"))
        .join(customers.set_index("CustomerID")[["CompanyName", "Country"]])
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )
    table["Share"] = table["Revenue"] / table["Revenue"].sum()
    table["CumulativeShare"] = table["Share"].cumsum()
    table.insert(0, "Rank", range(1, len(table) + 1))
    cols = ["Rank", "CustomerID", "CompanyName", "Country", "Orders",
            "FirstOrder", "LastOrder", "Revenue", "Share", "CumulativeShare"]
    return table[cols]


def concentration_metrics(ranked):
    """Summarize how concentrated revenue is across ranked customers.

    Args:
        ranked (pandas.DataFrame): Output of ``customer_revenue``.

    Returns:
        dict[str, float]: ``Customers`` (count), ``Revenue`` (total),
            ``Top1``, ``Top5``, and ``Top10`` (shares), ``Top20Pct``
            (share of the top 20% of customers), ``CustomersFor50``
            and ``CustomersFor80`` (fewest customers reaching that
            share), ``BottomHalf`` (share of the lower half), and
            ``HHI`` (Herfindahl index on a 0-10,000 scale).
    """
    share = ranked["Share"]
    cum = ranked["CumulativeShare"]
    n = len(ranked)
    return {
        "Customers": n,
        "Revenue": ranked["Revenue"].sum(),
        "Top1": share.iloc[0],
        "Top5": share.iloc[:5].sum(),
        "Top10": share.iloc[:TOP_N].sum(),
        "Top20Pct": share.iloc[:round(n * 0.2)].sum(),
        "CustomersFor50": int((cum < 0.5).sum() + 1),
        "CustomersFor80": int((cum < 0.8).sum() + 1),
        "BottomHalf": share.iloc[n // 2:].sum(),
        "HHI": (share ** 2).sum() * 10_000,
    }


def metrics_by_period(orders, customers):
    """Compute concentration metrics for each period in ``PERIODS``.

    Args:
        orders (pandas.DataFrame): Order rows from
            ``country_growth.load_order_revenue``.
        customers (pandas.DataFrame): The Customers sheet.

    Returns:
        pandas.DataFrame: One row per period, indexed by period label,
            with the columns from ``concentration_metrics``.
    """
    rows = {}
    for label, (start, end) in PERIODS.items():
        inside = orders
        if start:
            inside = inside[inside["OrderDate"] >= start]
        if end:
            inside = inside[inside["OrderDate"] < end]
        rows[label] = concentration_metrics(
            customer_revenue(inside, customers)
        )
    return pd.DataFrame.from_dict(rows, orient="index")


def plot_pareto(ranked, out_path):
    """Draw the top customers and the cumulative revenue curve.

    The left panel ranks the largest customers by share of revenue.
    The right panel plots cumulative share against customer rank for
    every customer. The two panels use separate axes rather than one
    dual-axis chart.

    Args:
        ranked (pandas.DataFrame): Output of ``customer_revenue``.
        out_path (pathlib.Path): PNG file to write.

    Returns:
        None. Writes the figure to ``out_path``.
    """
    metrics = concentration_metrics(ranked)
    n = len(ranked)

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12, 6.2), gridspec_kw={"width_ratios": [1, 1.15]}
    )
    fig.patch.set_facecolor(SURFACE)

    # Left: top customers as ranked horizontal bars, top N emphasized.
    top = ranked.head(BARS_SHOWN).iloc[::-1]
    colors = [BLUE if r <= TOP_N else GRAY for r in top["Rank"]]
    labels = [f"{name} ({country})" for name, country
              in zip(top["CompanyName"], top["Country"])]
    _style_axes(left)
    left.grid(axis="y", visible=False)
    left.grid(axis="x", color=GRID, linewidth=0.8)
    left.spines["bottom"].set_visible(False)
    left.barh(labels, top["Share"], color=colors, height=0.7)
    for y, share in enumerate(top["Share"]):
        left.text(share + 0.0015, y, f"{share:.1%}", va="center",
                  fontsize=8, color=INK_SECONDARY)
    left.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    left.set_xlim(0, top["Share"].max() * 1.18)
    left.tick_params(axis="y", labelsize=8.5, labelcolor=INK)
    left.set_title(f"Top {BARS_SHOWN} customers, share of revenue",
                   loc="left", fontsize=10.5, color=INK, pad=8)

    # Right: cumulative share by customer rank, with an equal-split
    # reference line and markers at the top N and the 80% point.
    _style_axes(right)
    ranks = ranked["Rank"]
    cum = ranked["CumulativeShare"]
    right.plot([0, n], [0, 1], color=BASELINE, linewidth=1,
               linestyle=(0, (4, 3)))
    right.text(n * 0.62, 0.5, "If every customer\nbought the same",
               fontsize=8, color=INK_MUTED, rotation=0)
    right.fill_between([0, *ranks[:TOP_N]], [0, *cum[:TOP_N]],
                       color=BLUE_LIGHT, linewidth=0)
    right.plot([0, *ranks], [0, *cum], color=BLUE, linewidth=2)

    k80 = metrics["CustomersFor80"]
    for rank, label in ((TOP_N, f"Top {TOP_N} customers\n"
                         f"= {metrics['Top10']:.0%} of revenue"),
                        (k80, f"{k80} customers\n= 80% of revenue")):
        value = cum.iloc[rank - 1]
        right.scatter([rank], [value], s=48, color=BLUE, zorder=3,
                      edgecolor=SURFACE, linewidth=2)
        right.annotate(label, (rank, value), xytext=(10, -26),
                       textcoords="offset points", fontsize=8.5,
                       color=INK)

    right.set_xlim(0, n + 1)
    right.set_ylim(0, 1.02)
    right.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    right.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    right.set_xlabel("Customers ranked by revenue (largest first)",
                     fontsize=8.5, color=INK_MUTED)
    right.set_title("Cumulative share of revenue", loc="left",
                    fontsize=10.5, color=INK, pad=8)

    fig.suptitle(
        "Customer concentration: the top 10 of 89 customers bring in "
        f"{metrics['Top10']:.0%} of revenue",
        x=0.015, ha="left", fontsize=13, color=INK, y=0.975,
    )
    fig.text(
        0.015, 0.905,
        "Revenue after discounts, all orders from 4 Jul 1996 to 6 May "
        "1998 (830 orders, $1.27M). 2 of 91 customers never ordered and "
        "are excluded.",
        fontsize=9, color=INK_SECONDARY,
    )
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.1, top=0.82,
                        wspace=0.18)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    """Run the analysis and write all outputs.

    Args:
        None.

    Returns:
        None. Writes CSVs to ``data/processed/`` and a PNG to
        ``figures/``, and prints the key numbers.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    orders = load_order_revenue(DATA_FILE)
    customers = pd.read_excel(DATA_FILE, sheet_name="Customers")

    ranked = customer_revenue(orders, customers)
    ranked.to_csv(PROCESSED_DIR / "customer_revenue.csv", index=False)

    periods = metrics_by_period(orders, customers)
    periods.to_csv(PROCESSED_DIR / "customer_concentration.csv",
                   index_label="Period")

    plot_pareto(ranked, FIGURES_DIR / "customer_pareto.png")

    never = sorted(set(customers["CustomerID"]) - set(ranked["CustomerID"]))
    print(f"Customers with orders: {len(ranked)} of {len(customers)}; "
          f"never ordered: {', '.join(never)}")
    pct = "{:.1%}".format
    print(periods.to_string(formatters={
        "Customers": "{:.0f}".format, "Revenue": "{:,.0f}".format,
        "Top1": pct, "Top5": pct, "Top10": pct, "Top20Pct": pct,
        "BottomHalf": pct, "HHI": "{:.0f}".format,
    }))
    print(ranked.head(15).to_string(index=False, formatters={
        "Revenue": "{:,.0f}".format, "Share": pct, "CumulativeShare": pct,
        "FirstOrder": "{:%Y-%m-%d}".format,
        "LastOrder": "{:%Y-%m-%d}".format,
    }))


if __name__ == "__main__":
    main()
