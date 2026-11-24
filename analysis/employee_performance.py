"""Sales and fulfillment performance per employee for Northwind.

Answers business question 6: revenue, order count, average order value,
and late-shipment rate for each employee. Writes a summary CSV and one
four-panel chart.

Run from the repository root:
    .venv/bin/python analysis/employee_performance.py
"""

import math

import matplotlib

matplotlib.use("Agg")  # Render to files only; no display needed.
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

# Shared loader, paths, and colors, so revenue is defined the same way
# in every analysis.
from country_growth import (
    BLUE, BLUE_LIGHT, DATA_FILE, FIGURES_DIR, GRID, INK, INK_MUTED,
    INK_SECONDARY, PROCESSED_DIR, SURFACE, _money, _style_axes,
    load_order_revenue,
)

Z_95 = 1.96  # Normal quantile for a 95% confidence interval.


def wilson_interval(late, shipped):
    """Return the 95% Wilson confidence interval for a late rate.

    The Wilson interval stays inside 0-1 and behaves well for the small
    counts here (1-10 late orders per employee).

    Args:
        late (int): Orders shipped after their required date.
        shipped (int): Orders shipped in total.

    Returns:
        tuple[float, float]: Lower and upper bounds as fractions.
    """
    if shipped == 0:
        return (math.nan, math.nan)
    p = late / shipped
    denom = 1 + Z_95 ** 2 / shipped
    center = (p + Z_95 ** 2 / (2 * shipped)) / denom
    half = (Z_95 / denom) * math.sqrt(
        p * (1 - p) / shipped + Z_95 ** 2 / (4 * shipped ** 2)
    )
    return (center - half, center + half)


def employee_metrics(path):
    """Compute the four performance metrics for each employee.

    An order is late when ``ShippedDate`` is after ``RequiredDate``.
    Unshipped orders count toward revenue and order count but not
    toward the late rate, because their outcome is not yet known.

    Args:
        path (pathlib.Path): Northwind Excel workbook.

    Returns:
        pandas.DataFrame: One row per employee, sorted by revenue
            (largest first), with ``EmployeeID``, ``Name``, ``Title``,
            ``Orders``, ``Revenue``, ``AvgOrderValue``, ``Shipped``,
            ``Late``, ``LateRate``, ``LateLow``, ``LateHigh`` (95%
            interval bounds), and ``RushShare`` (fraction of orders
            due 14 days after ordering).
    """
    sheets = pd.read_excel(path, sheet_name=["Orders", "Employees"])
    orders = sheets["Orders"][["OrderID", "EmployeeID", "OrderDate",
                               "RequiredDate", "ShippedDate"]]
    employees = sheets["Employees"]

    df = load_order_revenue(path)[["OrderID", "Revenue"]].merge(
        orders, on="OrderID", how="inner", validate="one_to_one"
    )
    df["Shipped"] = df["ShippedDate"].notna()
    df["Late"] = df["ShippedDate"] > df["RequiredDate"]  # NaT gives False.
    df["Rush"] = (df["RequiredDate"] - df["OrderDate"]).dt.days <= 14

    table = df.groupby("EmployeeID").agg(
        Orders=("OrderID", "nunique"),
        Revenue=("Revenue", "sum"),
        Shipped=("Shipped", "sum"),
        Late=("Late", "sum"),
        RushShare=("Rush", "mean"),
    )
    table["AvgOrderValue"] = table["Revenue"] / table["Orders"]
    table["LateRate"] = table["Late"] / table["Shipped"]
    bounds = [wilson_interval(l, s)
              for l, s in zip(table["Late"], table["Shipped"])]
    table["LateLow"], table["LateHigh"] = zip(*bounds)

    names = employees.set_index("EmployeeID")
    table = table.join(names[["Title"]])
    table["Name"] = names["FirstName"] + " " + names["LastName"]

    cols = ["Name", "Title", "Orders", "Revenue", "AvgOrderValue",
            "Shipped", "Late", "LateRate", "LateLow", "LateHigh",
            "RushShare"]
    return (table[cols].sort_values("Revenue", ascending=False)
            .reset_index())


def company_late_rate(table):
    """Return the late rate across all shipped orders.

    Args:
        table (pandas.DataFrame): Output of ``employee_metrics``.

    Returns:
        float: Late orders divided by shipped orders, as a fraction.
    """
    return table["Late"].sum() / table["Shipped"].sum()


def plot_scorecard(table, out_path):
    """Draw the four metrics as side-by-side panels sharing employee rows.

    Each metric has its own x-axis, so no panel needs a second scale.
    Employees are ordered by revenue in every panel. The late-rate
    panel shows 95% intervals and the company-wide rate for reference.

    Args:
        table (pandas.DataFrame): Output of ``employee_metrics``.
        out_path (pathlib.Path): PNG file to write.

    Returns:
        None. Writes the figure to ``out_path``.
    """
    rows = table.iloc[::-1].reset_index(drop=True)  # Largest at the top.
    y = list(range(len(rows)))
    labels = [f"{n}\n{t}" for n, t in zip(rows["Name"], rows["Title"])]
    overall = company_late_rate(table)

    fig, axes = plt.subplots(
        1, 4, figsize=(13, 6.4), sharey=True,
        gridspec_kw={"width_ratios": [1.15, 1, 1, 1.25]},
    )
    fig.patch.set_facecolor(SURFACE)

    bar_panels = [
        ("Revenue", "Revenue", _money, lambda v: f"${v / 1000:,.0f}k"),
        ("Orders", "Order count", None, lambda v: f"{v:,.0f}"),
        ("AvgOrderValue", "Average order value",
         lambda v, _: f"${v:,.0f}", lambda v: f"${v:,.0f}"),
    ]
    for ax, (col, title, tick_fmt, label_fmt) in zip(axes, bar_panels):
        _style_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.spines["bottom"].set_visible(False)
        ax.barh(y, rows[col], color=BLUE, height=0.62)
        for yi, v in zip(y, rows[col]):
            ax.text(v, yi, " " + label_fmt(v), va="center", fontsize=8,
                    color=INK_SECONDARY)
        ax.set_xlim(0, rows[col].max() * 1.32)
        if tick_fmt:
            ax.xaxis.set_major_formatter(FuncFormatter(tick_fmt))
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)

    # Late-rate panel: point estimate, 95% interval, company average.
    ax = axes[3]
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines["bottom"].set_visible(False)
    ax.axvline(overall, color=INK_MUTED, linewidth=1,
               linestyle=(0, (4, 3)), zorder=1)
    ax.text(overall, len(rows) - 0.35, f"All orders {overall:.1%}",
            ha="center", va="bottom", fontsize=8, color=INK_MUTED,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    ax.hlines(y, rows["LateLow"], rows["LateHigh"], color=BLUE_LIGHT,
              linewidth=3, zorder=2)
    ax.scatter(rows["LateRate"], y, s=48, color=BLUE, zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
    for yi, (rate, late, shipped, hi) in enumerate(zip(
            rows["LateRate"], rows["Late"], rows["Shipped"],
            rows["LateHigh"])):
        ax.text(hi + 0.006, yi, f"{rate:.1%} ({late:.0f}/{shipped:.0f})",
                va="center", fontsize=8, color=INK_SECONDARY)
    ax.set_xlim(0, rows["LateHigh"].max() + 0.11)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_title("Late-shipment rate (95% interval)", loc="left",
                 fontsize=10.5, color=INK, pad=8)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8.5, color=INK)
    axes[0].set_ylim(-0.6, len(rows) - 0.1)

    fig.suptitle(
        "Employee performance: revenue, orders, order value, and late "
        "shipments",
        x=0.012, ha="left", fontsize=13, color=INK, y=0.975,
    )
    fig.text(
        0.012, 0.9,
        "All orders from 4 Jul 1996 to 6 May 1998, sorted by revenue. "
        "Late = shipped after the required date; the 21 orders not yet "
        "shipped are left out of the late rate.\nWide intervals mean "
        "too few late orders to tell employees apart.",
        fontsize=9, color=INK_SECONDARY,
    )
    fig.subplots_adjust(left=0.165, right=0.985, bottom=0.06, top=0.8,
                        wspace=0.12)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    """Run the analysis and write all outputs.

    Args:
        None.

    Returns:
        None. Writes a CSV to ``data/processed/`` and a PNG to
        ``figures/``, and prints the key numbers.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    table = employee_metrics(DATA_FILE)
    table.to_csv(PROCESSED_DIR / "employee_performance.csv", index=False)
    plot_scorecard(table, FIGURES_DIR / "employee_performance.png")

    pct = "{:.1%}".format
    print(f"Company late rate: {company_late_rate(table):.1%} "
          f"({table['Late'].sum():.0f} of {table['Shipped'].sum():.0f} "
          f"shipped orders); unshipped: "
          f"{table['Orders'].sum() - table['Shipped'].sum():.0f}")
    print(table.to_string(index=False, formatters={
        "Revenue": "{:,.0f}".format, "AvgOrderValue": "{:,.0f}".format,
        "Shipped": "{:.0f}".format, "Late": "{:.0f}".format,
        "LateRate": pct, "LateLow": pct, "LateHigh": pct,
        "RushShare": pct,
    }))


if __name__ == "__main__":
    main()
