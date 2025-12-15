from api import (
    process_transactions,
    generate_all_insights,
    generate_custom_insight,
    export_analytics_json,
)

csv_path = "data/sample_transactions.csv"

df, analytics = process_transactions(csv_path)
print("Cleaned rows:", len(df))
print("Unique tickers:", analytics["unique_tickers"])
print("Total volume:", analytics["total_volume"])

print("\n--- JSON export preview ---")
print(export_analytics_json(csv_path)[:500])

print("\n--- All insights (local) ---")
ins = generate_all_insights(csv_path, api_provider="local")
print(ins["patterns"])
print(ins["risks"])

print("\n--- Custom insight ---")
print(generate_custom_insight(csv_path, "What is the biggest concentration risk?", api_provider="local"))
