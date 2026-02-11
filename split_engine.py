# split_engine.py

def split_bill(df, assignments):
    groups = {}
    for idx, row in df.iterrows():
        item_total = row["total"]
        assigned_groups = assignments.get(idx, [])
        
        if not assigned_groups: # Если никто не выбран, пропускаем позицию
            continue

        share = item_total / len(assigned_groups)
        for g in assigned_groups:
            groups[g] = groups.get(g, 0) + share
    return groups