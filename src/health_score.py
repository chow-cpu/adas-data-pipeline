def calculate_health_score(df, ml_outliers, zscore_outliers):
    total_rows = len(df)

    ml_rate = len(ml_outliers) / total_rows
    ml_score = max(0, 100 - (ml_rate * 200))

    zscore_rate = len(zscore_outliers) / total_rows
    zscore_score = max(0, 100 - (zscore_rate * 300))

    speed_std = df["speed_mps"].std()
    speed_mean = df["speed_mps"].mean()
    cv = (speed_std / speed_mean) * 100
    stability_score = max(0, 100 - cv)

    final_score = (ml_score * 0.5) + (zscore_score * 0.3) + (stability_score * 0.2)
    final_score = round(max(0, min(100, final_score)), 1)

    if final_score >= 75:
        status = "HEALTHY"
        color = "green"
    elif final_score >= 50:
        status = "WARNING"
        color = "orange"
    else:
        status = "CRITICAL"
        color = "red"

    return final_score, status, color