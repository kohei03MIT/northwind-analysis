"""Whether discounts increase order size or just give away revenue.

Answers business question 4. Each discounted order line is compared
with full-price lines of the same product at the same unit price, so
differences between products and list-price changes over time do not
distort the result. Also splits discount dollars by customer to show
how much goes to the largest customers. Writes two CSVs and one
two-panel chart.

The dataset has no cost data, so this measures revenue given away and
revenue gained, not profit.

Run from the repository root:
    .venv/bin/python analysis/discount_effect.py
"""

import matplotlib

matplotlib.use("Agg")  # Render to files only; no display needed.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

# Shared paths, colors, and styling from the first analysis.
from country_growth import (
    BLUE, BLUE_LIGHT, DATA_FILE, FIGURES_DIR, GRID, INK, INK_MUTED,
    INK_SECONDARY, PROCESSED_DIR, SURFACE, _money, _style_axes,
    load_order_revenue,
)
# Customer ranking from question 2, so "top 10" means the same customers
# here as on the customer concentration chart.
from customer_concentration import TOP_N, customer_revenue

BANDS = ["5%", "10%", "15%", "20%", "25%"]
ALL = "All discounts"
N_BOOT = 1000   # Bootstrap resamples for the 95% intervals.
SEED = 561      # Fixed so the intervals are reproducible.

# The cell that defines "the same product": a product sold at one unit
# price. Prices rose for every product during the period, so comparing
# across prices would mix in that change.
CELL = ["ProductID", "UnitPrice"]


def load_lines(path):
    """Load order lines with gross, net, and given-away revenue.

    Only the Orders Details sheet is used; the hand-made price columns
    in the Product sheet are not.

    Args:
        path (pathlib.Path): Northwind Excel workbook.

    Returns:
        pandas.DataFrame: One row per order line with the original
            columns plus ``Discounted`` (bool), ``Band`` (discount band
            label; "0%" for full price), ``Gross`` (UnitPrice x
            Quantity), ``Net`` (Gross after discount), and ``GivenAway``
            (Gross minus Net).
    """
    lines = pd.read_excel(path, sheet_name="Orders Details")
    lines["Discounted"] = lines["Discount"] > 0

    # Eight lines carry odd discounts of 1-6%; they join the 5% band.
    pct = (lines["Discount"] * 100).round().astype(int)
    lines["Band"] = np.where(
        pct == 0, "0%",
        np.where(pct <= 6, "5%", pct.astype(str) + "%"),
    )

    lines["Gross"] = lines["UnitPrice"] * lines["Quantity"]
    lines["Net"] = lines["Gross"] * (1 - lines["Discount"])
    lines["GivenAway"] = lines["Gross"] - lines["Net"]
    return lines


def attach_baseline(lines):
    """Add the full-price benchmark quantity to each discounted line.

    The benchmark is the mean quantity of full-price lines in the same
    product and unit-price cell. Discounted lines in cells with no
    full-price line have no benchmark and are dropped.

    Args:
        lines (pandas.DataFrame): Output of ``load_lines`` or a
            resample of it.

    Returns:
        pandas.DataFrame: The comparable discounted lines, with
            ``BaseQty`` (benchmark quantity), ``BaseRevenue`` (unit
            price x benchmark quantity, i.e. a typical full-price line),
            and ``RevenueAtBase`` (net revenue had the line kept the
            benchmark quantity, used for the break-even lift).
    """
    base = (lines[~lines["Discounted"]].groupby(CELL)["Quantity"].mean()
            .rename("BaseQty"))
    disc = lines[lines["Discounted"]].join(base, on=CELL)
    disc = disc.dropna(subset=["BaseQty"])
    disc["BaseRevenue"] = disc["UnitPrice"] * disc["BaseQty"]
    disc["RevenueAtBase"] = disc["BaseRevenue"] * (1 - disc["Discount"])
    return disc


def effect(disc):
    """Measure the quantity and revenue effect for a set of lines.

    Args:
        disc (pandas.DataFrame): Output of ``attach_baseline``.

    Returns:
        dict[str, float]: ``Lines``; ``QtyLift`` (actual quantity over
            benchmark quantity, minus 1); ``BreakEvenLift`` (lift needed
            for net revenue to match a full-price line); ``RevenueVsFull``
            (net revenue over benchmark full-price revenue, minus 1);
            ``Net`` and ``GivenAway`` (dollars).
    """
    return {
        "Lines": len(disc),
        "QtyLift": disc["Quantity"].sum() / disc["BaseQty"].sum() - 1,
        "BreakEvenLift": (disc["BaseRevenue"].sum()
                          / disc["RevenueAtBase"].sum() - 1),
        "RevenueVsFull": disc["Net"].sum() / disc["BaseRevenue"].sum() - 1,
        "Net": disc["Net"].sum(),
        "GivenAway": disc["GivenAway"].sum(),
    }


def effects_by_band(disc):
    """Apply ``effect`` to each discount band and to all bands together.

    Args:
        disc (pandas.DataFrame): Output of ``attach_baseline``.

    Returns:
        pandas.DataFrame: One row per band in ``BANDS`` order, then an
            ``ALL`` row, with the columns returned by ``effect``.
    """
    rows = {b: effect(disc[disc["Band"] == b]) for b in BANDS}
    rows[ALL] = effect(disc)
    return pd.DataFrame.from_dict(rows, orient="index")


def bootstrap_intervals(lines):
    """Estimate 95% intervals for quantity lift and revenue effect.

    Full-price and discounted lines are resampled separately, with
    replacement, so both the benchmark and the discounted quantities
    vary between resamples.

    Args:
        lines (pandas.DataFrame): Output of ``load_lines``.

    Returns:
        pandas.DataFrame: Indexed like ``effects_by_band``, with
            ``QtyLiftLow``, ``QtyLiftHigh``, ``RevenueVsFullLow``, and
            ``RevenueVsFullHigh`` (2.5th and 97.5th percentiles).
    """
    rng = np.random.default_rng(SEED)
    full = lines[~lines["Discounted"]]
    discounted = lines[lines["Discounted"]]
    draws = []
    for _ in range(N_BOOT):
        sample = pd.concat([
            full.iloc[rng.integers(0, len(full), len(full))],
            discounted.iloc[rng.integers(0, len(discounted),
                                         len(discounted))],
        ])
        table = effects_by_band(attach_baseline(sample))
        draws.append(table[["QtyLift", "RevenueVsFull"]])
    stacked = pd.concat(draws, keys=range(N_BOOT))
    grouped = stacked.groupby(level=1)
    out = pd.DataFrame({
        "QtyLiftLow": grouped["QtyLift"].quantile(0.025),
        "QtyLiftHigh": grouped["QtyLift"].quantile(0.975),
        "RevenueVsFullLow": grouped["RevenueVsFull"].quantile(0.025),
        "RevenueVsFullHigh": grouped["RevenueVsFull"].quantile(0.975),
    })
    return out.reindex(BANDS + [ALL])


def discount_by_customer(lines, path):
    """Split revenue given away by customer, ranked by revenue.

    Uses every discounted line, including the few without a full-price
    match, so the total equals all revenue given away.

    Args:
        lines (pandas.DataFrame): Output of ``load_lines``.
        path (pathlib.Path): Northwind Excel workbook.

    Returns:
        pandas.DataFrame: One row per customer with at least one order,
            in revenue-rank order, with ``Rank``, ``CustomerID``,
            ``CompanyName``, ``Country``, ``Revenue``, ``RevenueShare``,
            ``DiscountedLines``, ``Gross`` (revenue before discounts),
            ``GivenAway``, ``DiscountShare`` (share of all revenue given
            away), ``EffectiveDiscount`` (GivenAway over Gross), and
            ``ShareGap`` (DiscountShare minus RevenueShare; positive
            means more than its share).
    """
    sheets = pd.read_excel(path, sheet_name=["Orders", "Customers"])
    with_customer = lines.merge(
        sheets["Orders"][["OrderID", "CustomerID"]], on="OrderID",
        how="inner", validate="many_to_one",
    )
    per_customer = with_customer.groupby("CustomerID").agg(
        DiscountedLines=("Discounted", "sum"),
        Gross=("Gross", "sum"),
        GivenAway=("GivenAway", "sum"),
    )

    ranked = customer_revenue(load_order_revenue(path), sheets["Customers"])
    table = ranked[["Rank", "CustomerID", "CompanyName", "Country",
                    "Revenue", "Share"]].rename(
        columns={"Share": "RevenueShare"}
    ).join(per_customer, on="CustomerID")

    table["DiscountShare"] = table["GivenAway"] / table["GivenAway"].sum()
    table["EffectiveDiscount"] = table["GivenAway"] / table["Gross"]
    table["ShareGap"] = table["DiscountShare"] - table["RevenueShare"]
    table["DiscountedLines"] = table["DiscountedLines"].astype(int)
    return table


def plot_discount_effect(summary, out_path):
    """Draw quantity lift against break-even lift, and revenue given away.

    Left panel: for each discount band, the measured quantity lift with
    its 95% interval, next to the lift needed just to keep revenue
    level with a full-price line. Right panel: dollars given away per
    band. The panels have separate axes.

    Args:
        summary (pandas.DataFrame): ``effects_by_band`` joined with
            ``bootstrap_intervals``.
        out_path (pathlib.Path): PNG file to write.

    Returns:
        None. Writes the figure to ``out_path``.
    """
    bands = summary.loc[BANDS]
    x = np.arange(len(BANDS))
    overall = summary.loc[ALL]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    fig.patch.set_facecolor(SURFACE)

    # Left: measured lift (dot + interval) vs break-even lift (dash).
    _style_axes(left)
    left.vlines(x, bands["QtyLiftLow"], bands["QtyLiftHigh"],
                color=BLUE_LIGHT, linewidth=6, zorder=2)
    left.scatter(x, bands["QtyLift"], s=70, color=BLUE, zorder=4,
                 edgecolor=SURFACE, linewidth=2,
                 label="Measured extra quantity (95% interval)")
    left.scatter(x, bands["BreakEvenLift"], s=420, marker="_",
                 color=INK, linewidth=2.5, zorder=3,
                 label="Extra quantity needed to break even on revenue")
    for xi, (lift, need) in enumerate(zip(bands["QtyLift"],
                                          bands["BreakEvenLift"])):
        left.text(xi + 0.13, lift, f"+{lift:.0%}", va="center",
                  fontsize=8.5, color=INK_SECONDARY)
        left.text(xi - 0.13, need, f"+{need:.0%}", va="center",
                  ha="right", fontsize=8.5, color=INK)
    left.set_xticks(x)
    left.set_xticklabels([f"{b} off" for b in BANDS], fontsize=9,
                         color=INK)
    left.set_xlim(-0.6, len(BANDS) - 0.4)
    left.set_ylim(0, max(bands["QtyLiftHigh"].max(),
                             bands["BreakEvenLift"].max()) + 0.08)
    left.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    left.legend(loc="upper left", frameon=False, fontsize=8.5,
                labelcolor=INK_SECONDARY)
    left.set_title(
        "Quantity per line vs full-price lines of the same product",
        loc="left", fontsize=10.5, color=INK, pad=8,
    )

    # Right: revenue given away per band.
    _style_axes(right)
    right.bar(x, bands["GivenAway"], color=BLUE, width=0.6)
    for xi, v in enumerate(bands["GivenAway"]):
        right.text(xi, v, f"${v / 1000:,.1f}k", ha="center", va="bottom",
                   fontsize=8.5, color=INK_SECONDARY)
    right.set_xticks(x)
    right.set_xticklabels([f"{b} off" for b in BANDS], fontsize=9,
                          color=INK)
    right.yaxis.set_major_formatter(FuncFormatter(_money))
    right.set_ylim(0, bands["GivenAway"].max() * 1.15)
    right.set_title("Revenue given away by discounts", loc="left",
                    fontsize=10.5, color=INK, pad=8)

    fig.suptitle(
        "Discounts come with about "
        f"{overall['QtyLift']:.0%} more quantity, whether 5% or 25% off",
        x=0.012, ha="left", fontsize=13, color=INK, y=0.975,
    )
    fig.text(
        0.012, 0.885,
        "Each discounted line is compared with full-price lines of the same "
        "product at the same unit price (all orders, Jul 1996 – May 1998).\n"
        "Break even = the extra quantity needed for net revenue to match a "
        "full-price line. No cost data, so this is revenue, not profit.",
        fontsize=9, color=INK_SECONDARY,
    )
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.08, top=0.78,
                        wspace=0.18)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    """Run the analysis and write all outputs.

    Args:
        None.

    Returns:
        None. Writes two CSVs to ``data/processed/`` and a PNG to
        ``figures/``, and prints the key numbers.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    lines = load_lines(DATA_FILE)
    disc = attach_baseline(lines)
    summary = effects_by_band(disc).join(bootstrap_intervals(lines))
    summary.to_csv(PROCESSED_DIR / "discount_effect.csv", index_label="Band")
    plot_discount_effect(summary, FIGURES_DIR / "discount_effect.png")

    by_customer = discount_by_customer(lines, DATA_FILE)
    by_customer.to_csv(PROCESSED_DIR / "discount_by_customer.csv",
                       index=False)

    full = lines[~lines["Discounted"]]
    print(f"Lines: {len(lines)}; discounted: {lines['Discounted'].sum()} "
          f"({lines['Discounted'].mean():.1%}); comparable discounted "
          f"lines: {len(disc)}")
    print(f"Gross ${lines['Gross'].sum():,.0f}, net ${lines['Net'].sum():,.0f},"
          f" given away ${lines['GivenAway'].sum():,.0f} "
          f"({lines['GivenAway'].sum() / lines['Net'].sum():.1%} of net "
          f"revenue)")
    print(f"Mean quantity per line: full price {full['Quantity'].mean():.1f}, "
          f"discounted {lines[lines['Discounted']]['Quantity'].mean():.1f}")
    pct = "{:+.1%}".format
    print(summary.to_string(formatters={
        "Lines": "{:.0f}".format, "QtyLift": pct, "BreakEvenLift": pct,
        "RevenueVsFull": pct, "Net": "{:,.0f}".format,
        "GivenAway": "{:,.0f}".format, "QtyLiftLow": pct,
        "QtyLiftHigh": pct, "RevenueVsFullLow": pct,
        "RevenueVsFullHigh": pct,
    }))

    top = by_customer.head(TOP_N)
    others = by_customer.iloc[TOP_N:]
    print(f"\nTop {TOP_N} customers by revenue: "
          f"{top['RevenueShare'].sum():.1%} of revenue, "
          f"{top['DiscountShare'].sum():.1%} of revenue given away "
          f"(${top['GivenAway'].sum():,.0f} of "
          f"${by_customer['GivenAway'].sum():,.0f})")
    # Pooled rate: total given away over total gross, so customers with
    # no discounts still count in the denominator.
    rate = lambda t: t["GivenAway"].sum() / t["Gross"].sum()
    print(f"Effective discount rate: top {TOP_N} {rate(top):.1%}, "
          f"other customers {rate(others):.1%}, all {rate(by_customer):.1%}")
    share = "{:.1%}".format
    print(top.to_string(index=False, formatters={
        "Revenue": "{:,.0f}".format, "RevenueShare": share,
        "Gross": "{:,.0f}".format,
        "GivenAway": "{:,.0f}".format, "DiscountShare": share,
        "EffectiveDiscount": share, "ShareGap": "{:+.1%}".format,
    }))


if __name__ == "__main__":
    main()
