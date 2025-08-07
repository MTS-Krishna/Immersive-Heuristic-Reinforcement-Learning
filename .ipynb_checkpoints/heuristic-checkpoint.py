def heuristic_rule(row):
    if row['cpu_percent'] > 85 and row['memory_percent'] > 63:
        return 1  # urgent alert
    elif row['cpu_percent'] > 65:
        return 0  # moderate attention
    else:
        return 2  # normal
