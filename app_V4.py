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

DOMAIN_LABELS = {
    "DM": "Demography (DM)",
    "AE": "Adverse Events (AE)",
    "DS": "Disposition (DS)",
    "LB": "Laboratory Measurements (LB)",
    "VS": "Vital Signs (VS)"
}

def load_domain(domain_name):
    file_path = os.path.join(DATA_PATH, f"{domain_name.lower()}.sas7bdat")
    if not os.path.exists(file_path): return None
    try:
        df, _ = pyreadstat.read_sas7bdat(file_path)
        if "USUBJID" in df.columns:
            df["USUBJID"] = df["USUBJID"].astype(str).str.strip()
        dm_path = os.path.join(DATA_PATH, "dm.sas7bdat")
        if domain_name.upper() != "DM" and os.path.exists(dm_path):
            dm_df, _ = pyreadstat.read_sas7bdat(dm_path)
            dm_df["USUBJID"] = dm_df["USUBJID"].astype(str).str.strip()
            df = pd.merge(df, dm_df[["USUBJID", "ARM", "SEX"]], on="USUBJID", how="left")
        return df
    except Exception as e:
        return None

@app.route("/")
def index():
    if not os.path.exists(DATA_PATH): os.makedirs(DATA_PATH, exist_ok=True)
    sequence = ["DM", "AE", "DS", "LB", "VS"]
    domains = [{"id": d, "label": DOMAIN_LABELS[d]} for d in sequence if os.path.exists(os.path.join(DATA_PATH, f"{d.lower()}.sas7bdat"))]
    return render_template("dashboard.html", domains=domains)

@app.route("/get_metadata/<domain>")
def get_metadata(domain):
    df = load_domain(domain)
    if df is None: return jsonify({"error": "Not found"})
    return jsonify({
        "columns": list(df.columns),
        "numeric_cols": list(df.select_dtypes(include=[np.number]).columns)
    })

@app.route("/get_analysis/<domain>")
def get_analysis(domain):
    x, y = request.args.get('x'), request.args.get('y')
    chart_type = request.args.get('type', 'box')
    df = load_domain(domain)
    if df is None: return jsonify({"error": "Data error"})

    summary_html = ""
    if y and y in df.columns:
        df[y] = pd.to_numeric(df[y], errors='coerce')
        summary = df.groupby(x if x else 'ARM')[y].describe().round(2).reset_index()
        summary_html = summary.to_html(classes='summary-table', index=False)

    figs = []
    if domain.upper() == "DS":
        ds_col = "DSDECOD" if "DSDECOD" in df.columns else df.columns[0]
        fig = px.histogram(df, x=ds_col, color="ARM", barmode="group", title="Subject Disposition Status")
        figs.append(fig)
    elif chart_type == "line":
        line_df = df.groupby([x, 'ARM'])[y].mean().reset_index()
        figs.append(px.line(line_df, x=x, y=y, color="ARM", markers=True))
    elif chart_type == "box":
        figs.append(px.box(df, x=x, y=y, color="ARM", points="outliers"))
    elif chart_type == "pie":
        figs.append(px.pie(df, names=x))
    elif chart_type == "histogram":
        figs.append(px.histogram(df, x=x, color="ARM", barmode="group"))

    for f in figs: f.update_layout(template="plotly_white")
    
    return jsonify({"plots": [pio.to_json(f) for f in figs], "table": summary_html})

if __name__ == "__main__":
    app.run(debug=True, port=5000)