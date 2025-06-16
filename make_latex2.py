import pandas as pd

# Load the CSV
df = pd.read_csv('Madu_Results.csv')

# Display names for models
model_display = {
    'svd': 'SVT',
    'dft': 'DFT',
    'median': 'Median',
    'Series2': 'Serial',
    'Ensemble2': 'Ensemble',
    'Madu2': 'MADU'
}
model_keys = list(model_display.keys())

# Display names for categories
category_display = {
    'baseline': 'Baseline',
    'lowFramerate': 'Low Framerate',
    'thermal': 'Thermal',
    'badWeather': 'Bad Weather',
    'dynamicBackground': 'Dynamic Background',
    'shadow': 'Shadow',
    'nightVideos': 'Night Videos',
    'turbulence': 'Turbulence',
    'cameraJitter': 'Camera Jitter'
}

def pretty_cat(name):
    return category_display.get(name, name.title())

# Clean video names for low framerate
def clean_video_name(name, cat):
    if cat == 'lowFramerate' and '_' in name:
        return name.split('_')[0]
    return name

# Gather data rows
lines = []
lines.append(r'\begin{table*}[t]')
lines.append(r'\centering')
lines.append(r'\caption{F1 scores across all CDNet2014 categories}')
lines.append(r'\begin{tabular}{ll' + 'c'*len(model_display) + '}')
lines.append(r'\toprule')
lines.append('Category & Video & ' + ' & '.join(model_display.values()) + r' \\')
lines.append(r'\midrule')

# Collect all values for global average
all_model_values = {m: [] for m in model_display.values()}

# Process each category
for cat in sorted(df['Category Name'].unique()):
    subdf = df[df['Category Name'] == cat]
    vids = sorted(subdf['Category Video'].unique())
    
    # Pivot table per video
    pivot = subdf.pivot_table(index='Category Video', columns='Model Name', values='F1 (0.5)', aggfunc='mean')
    pivot = pivot.reindex(columns=model_keys).rename(columns=model_display)
    
    cat_label = pretty_cat(cat)
    for i, vid in enumerate(vids):
        clean_vid = clean_video_name(vid, cat)
        row = pivot.loc[vid]

        # Determine the best model in the row
        numeric = {c: float(v) for c, v in row.items() if pd.notna(v)}
        best = max(numeric.values()) if numeric else None

        cells = []
        for col in model_display.values():
            v = row.get(col, 'x')
            if pd.notna(v):
                v = f"{float(v):.2f}"
                if best is not None and float(v) == best:
                    v = f'\\textbf{{{v}}}'
                all_model_values[col].append(float(v.strip('\\textbf{}')))
            else:
                v = 'x'
            cells.append(v)

        prefix = rf'\multirow{{{len(vids)}}}{{*}}{{{cat_label}}}' if i == 0 else ''
        lines.append(f'{prefix} & {clean_vid} & ' + ' & '.join(cells) + r' \\')

    lines.append(r'\midrule')

# Calculate global averages
averages = {}
for col in model_display.values():
    vals = all_model_values[col]
    averages[col] = round(sum(vals) / len(vals), 2) if vals else 'x'

# Bold only the best average
numeric_avgs = {k: v for k, v in averages.items() if isinstance(v, float)}
best_avg = max(numeric_avgs.values()) if numeric_avgs else None

avg_row = []
for col in model_display.values():
    v = averages[col]
    if v != 'x' and best_avg is not None and v == best_avg:
        avg_row.append(f'\\textbf{{{v:.2f}}}')
    else:
        avg_row.append(f'{v:.2f}' if isinstance(v, float) else 'x')

lines.append(r'\textbf{Average} & & ' + ' & '.join(avg_row) + r' \\')
lines.append(r'\bottomrule')
lines.append(r'\label{tab:all_f1}')
lines.append(r'\end{tabular}')
lines.append(r'\end{table*}')

# Write to file
with open('madu_all_categories_table.tex', 'w') as out:
    out.write('\n'.join(lines))

print("✅ Generated 'madu_all_categories_table.tex'")
