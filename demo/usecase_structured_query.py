"""
Demo: Structured Data Query
=============================
Query CSV/Excel data using natural patterns — SUM, GROUP BY, FILTER.

Usage:
    python demo/usecase_structured_query.py
"""

import csv
from collections import defaultdict


def load_csv(path: str) -> list[dict]:
    """Load CSV into list of dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))


def query_total(data: list[dict], column: str) -> float:
    """SUM a column."""
    return sum(float(row[column]) for row in data)


def query_group_by(data: list[dict], group_col: str, sum_col: str) -> dict:
    """GROUP BY and SUM."""
    groups = defaultdict(float)
    for row in data:
        groups[row[group_col]] += float(row[sum_col])
    return dict(sorted(groups.items(), key=lambda x: x[1], reverse=True))


def query_filter(data: list[dict], column: str, value: str) -> list[dict]:
    """Filter rows by column value."""
    return [row for row in data if row[column] == value]


def query_count(data: list[dict], column: str, value: str) -> int:
    """Count rows matching condition."""
    return len(query_filter(data, column, value))


def query_max(data: list[dict], column: str) -> dict:
    """Find row with max value."""
    return max(data, key=lambda x: float(x[column]))


def query_min(data: list[dict], column: str) -> dict:
    """Find row with min value."""
    return min(data, key=lambda x: float(x[column]))


def query_avg(data: list[dict], column: str) -> float:
    """Average of a column."""
    values = [float(row[column]) for row in data]
    return sum(values) / len(values)


def main():
    print("=" * 60)
    print("DEMO: Structured Data Query")
    print("=" * 60)
    print()

    # Load data
    data = load_csv("demo/sample_sales.csv")
    print(f"Loaded: {len(data)} rows from sample_sales.csv")
    print(f"Columns: {list(data[0].keys())}")
    print()

    # Show data
    print("Data preview:")
    print(f"{'Date':<12} {'Vendor':<12} {'Product':<12} {'Qty':>5} {'Total':>10} {'Status':<8}")
    print("-" * 65)
    for row in data:
        print(f"{row['Date']:<12} {row['Vendor']:<12} {row['Product']:<12} {row['Quantity']:>5} {row['Total']:>10} {row['Status']:<8}")

    # Queries — same as dq.ask() will do
    print("\n" + "=" * 60)
    print("NATURAL LANGUAGE QUERIES (exact computation)")
    print("=" * 60)

    # Q1: What is the total amount?
    total = query_total(data, "Total")
    print(f"\nQ: What is the total amount?")
    print(f"A: {total:,.2f}  (SUM of Total column)")

    # Q2: Which vendor has the highest sales?
    by_vendor = query_group_by(data, "Vendor", "Total")
    top_vendor = list(by_vendor.items())[0]
    print(f"\nQ: Which vendor has the highest sales?")
    print(f"A: {top_vendor[0]} ({top_vendor[1]:,.2f})")
    for vendor, amount in by_vendor.items():
        print(f"   {vendor:<12} {amount:>12,.2f}")

    # Q3: How many invoices are overdue?
    overdue_count = query_count(data, "Status", "Overdue")
    overdue_rows = query_filter(data, "Status", "Overdue")
    overdue_total = sum(float(r["Total"]) for r in overdue_rows)
    print(f"\nQ: How many invoices are overdue?")
    print(f"A: {overdue_count} invoices totaling {overdue_total:,.2f}")

    # Q4: Average order value?
    avg = query_avg(data, "Total")
    print(f"\nQ: What is the average order value?")
    print(f"A: {avg:,.2f}")

    # Q5: Revenue by region?
    by_region = query_group_by(data, "Region", "Total")
    print(f"\nQ: Show me revenue by region")
    print(f"A:")
    for region, amount in by_region.items():
        print(f"   {region:<8} {amount:>12,.2f}")

    # Q6: Largest single order?
    largest = query_max(data, "Total")
    print(f"\nQ: What is the largest single order?")
    print(f"A: {largest['Vendor']} - {largest['Product']} - {float(largest['Total']):,.2f} ({largest['Date']})")

    # Q7: Pending orders?
    pending = query_filter(data, "Status", "Pending")
    print(f"\nQ: Show me all pending orders")
    print(f"A: {len(pending)} pending orders:")
    for row in pending:
        print(f"   {row['Date']} | {row['Vendor']:<12} | {row['Product']:<12} | {float(row['Total']):>10,.2f}")

    # Q8: Revenue by product?
    by_product = query_group_by(data, "Product", "Total")
    print(f"\nQ: Revenue by product?")
    print(f"A:")
    for product, amount in by_product.items():
        print(f"   {product:<12} {amount:>12,.2f}")

    print(f"\n--- All queries use exact computation, not LLM guessing ---")
    print("\nDone!")


if __name__ == "__main__":
    main()
