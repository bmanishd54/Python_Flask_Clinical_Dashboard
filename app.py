from flask import Flask, render_template, jsonify, request, url_for
from flask.json.provider import DefaultJSONProvider
import pandas as pd
import pyreadstat
import plotly.express as px
import plotly.io as pio
import os
import numpy as np

class CustomJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        from datetime import date, datetime
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)

app = Flask(__name__)
app.json = CustomJSONProvider(app)

DATA_PATH = "./SDTM_SAS/"

DOMAINS = {
    "SDTM": {
        "DM": "Demography (DM)", 
        "AE": "Adverse Events (AE)", 
        "DS": "Disposition (DS)", 
        "LB": "Laboratory Measurements (LB)", 
        "VS": "Vital Signs (VS)"
    },
    "ADAM": {
        "ADSL": "Subject Level Analysis (ADSL)",
        "ADAE": "Adverse Event Analysis (ADAE)"
    }
}

def load_data(file_name):
    path = os.path.join(DATA_PATH, f"{file_name.lower()}.sas7bdat")
    if not os.path.exists(path): return None
    try:
        df, _ = pyreadstat.read_sas7bdat(path)
        if "USUBJID" in df.columns:
            df["USUBJID"] = df["USUBJID"].astype(str).str.strip()
        
        dm_path = os.path.join(DATA_PATH, "dm.sas7bdat")
        if file_name.upper() in ["AE", "DS", "LB", "VS"] and os.path.exists(dm_path):
            dm_df, _ = pyreadstat.read_sas7bdat(dm_path)
            dm_df["USUBJID"] = dm_df["USUBJID"].astype(str).str.strip()
            df = pd.merge(df, dm_df[["USUBJID", "ARM", "SEX"]], on="USUBJID", how="left")
        return df
    except: return None

def get_clinical_summary(df, group_cols, y_col):
    try:
        stats = df.groupby(group_cols)[y_col].agg(['count', 'mean', 'std', 'median', 'min', 'max']).round(2)
        stats['std'] = stats['std'].fillna(0)
        stats['Mean (SD)'] = stats['mean'].astype(str) + " (" + stats['std'].astype(str) + ")"
        stats['Range (Min - Max)'] = stats['min'].astype(str) + " - " + stats['max'].astype(str)
        stats = stats[['count', 'Mean (SD)', 'median', 'Range (Min - Max)']].reset_index()
        stats.rename(columns={'count': 'N', 'median': 'Median'}, inplace=True)
        return stats.to_html(classes='summary-table', index=False)
    except Exception as e:
        return f"<p style='color:red;'>Unable to generate summary.</p>"

@app.route("/")
def index():
    if not os.path.exists(DATA_PATH): os.makedirs(DATA_PATH, exist_ok=True)
    return render_template("dashboard.html", domains=DOMAINS)

@app.route("/get_metadata/<dataset>")
def get_metadata(dataset):
    df = load_data(dataset)
    if df is None: return jsonify({"error": "Not found"})
    return jsonify({
        "columns": list(df.columns),
        "numeric_cols": list(df.select_dtypes(include=[np.number]).columns)
    })

@app.route("/get_analysis/<dataset>")
def get_analysis(dataset):
    x = request.args.get('x')
    y = request.args.get('y')
    grp = request.args.get('grp')
    chart_type = request.args.get('type', 'box')
    
    df = load_data(dataset)
    if df is None: return jsonify({"error": "Data error"})

    if grp == "NONE": grp = None
    elif not grp or grp not in df.columns:
        grp = next((c for c in ["TRTP", "TRTA", "ARM"] if c in df.columns), None)

    summary_html = ""
    figs = []
    param_col = next((c for c in [f"{dataset.upper()}TESTCD", "PARAMCD"] if c in df.columns), None)

    if chart_type in ["box", "line"] and y and y in df.columns and y in df.select_dtypes(include=[np.number]).columns:
        df[y] = pd.to_numeric(df[y], errors='coerce')
        plot_df = df.copy()

        if param_col:
            top_params = plot_df[param_col].value_counts().nlargest(6).index
            plot_df = plot_df[plot_df[param_col].isin(top_params)]

        group_cols = [c for c in [param_col, grp, x] if c and c in plot_df.columns]
        if not group_cols and grp: group_cols = [grp]
        if group_cols: summary_html = get_clinical_summary(plot_df, group_cols, y)

        if chart_type == "box":
            if param_col:
                fig = px.box(plot_df, x=x, y=y, color=grp, facet_col=param_col, facet_col_wrap=3, points="outliers")
                fig.update_yaxes(matches=None) 
            else:
                fig = px.box(plot_df, x=x, y=y, color=grp, points="outliers")
            figs.append(fig)

        elif chart_type == "line":
            sort_col = "AVISITN" if "AVISITN" in plot_df.columns else ("VISITNUM" if "VISITNUM" in plot_df.columns else None)
            group_vars = [c for c in [sort_col, x, grp, param_col] if c]
            
            line_df = plot_df.groupby(group_vars)[y].mean().reset_index()
            if sort_col: line_df = line_df.sort_values(sort_col)

            if param_col:
                fig = px.line(line_df, x=x, y=y, color=grp, facet_col=param_col, facet_col_wrap=3, markers=True)
                fig.update_yaxes(matches=None)
            else:
                fig = px.line(line_df, x=x, y=y, color=grp, markers=True)
            figs.append(fig)

    else:
        target_col = x if x else ("AEDECOD" if "AEDECOD" in df.columns else df.columns[0])
        counts = df.groupby([grp, target_col]).size().reset_index(name='Count') if grp else df.groupby([target_col]).size().reset_index(name='Count')
        summary_html = counts.sort_values("Count", ascending=False).to_html(classes='summary-table', index=False)
        
        if chart_type == "pie": 
            figs.append(px.pie(df, names=target_col, title=f"Proportion of {target_col}"))
        else: 
            figs.append(px.bar(counts.head(20), x=target_col, y="Count", color=grp, barmode="group", title=f"Frequency of {target_col}"))

    for f in figs: 
        f.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        f.update_layout(template="plotly_white", height=700 if param_col else 500)
    
    return jsonify({"plots": [pio.to_json(f) for f in figs], "table": summary_html})

if __name__ == "__main__":
    app.run(debug=True, port=5000)