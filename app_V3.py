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

# Labels updated to show both Full Name and Short Code
DOMAIN_LABELS = {
    "AE": "Adverse Events (AE)",
    "CM": "Concomitant Medication (CM)",
    "DM": "Demography (DM)",
    "DS": "Disposition (DS)",
    "EG": "Electrocardiogram (EG)",
    "EX": "Exposure (EX)",
    "LB": "Laboratory Measurements (LB)",
    "MH": "Medical History (MH)",
    "VS": "Vital Signs (VS)",
    "SV": "Subject Visits (SV)",
    "TS": "Trial Design (TS)",
    "IS": "Immunogenicity Specimen (IS)",
    "QS": "Questionnaires (QS)"
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
        print(f"Error loading {domain_name}: {e}")
        return None

@app.route("/")
def index():
    if not os.path.exists(DATA_PATH): os.makedirs(DATA_PATH, exist_ok=True)
    raw_domains = sorted([f.split(".")[0].upper() for f in os.listdir(DATA_PATH) if f.endswith(".sas7bdat")])
    domains = [{"id": d, "label": DOMAIN_LABELS.get(d, f"{d} ({d})")} for d in raw_domains]
    return render_template("dashboard.html", domains=domains)

@app.route("/get_plots/<domain>")
def get_plots(domain):
    view_mode = request.args.get('mode', 'mean')
    df = load_domain(domain)
    domain_label = DOMAIN_LABELS.get(domain.upper(), f"{domain.upper()} ({domain.upper()})")
    
    if df is None or df.empty: return jsonify({"error": "Data not found"})

    figs, titles = [], []
    res_col = f"{domain.upper()}STRESN" if f"{domain.upper()}STRESN" in df.columns else None
    test_col = f"{domain.upper()}TESTCD"
    EPHICACY_PALETTE = ["#004a99", "#10b981", "#f59e0b", "#6366f1", "#ec4899"]

    try:
        if domain.upper() in ["LB", "VS", "EG"]:
            titles = [f"{domain_label} Analysis", f"{domain_label} Flag Distribution"]
            if res_col and test_col in df.columns:
                df[res_col] = pd.to_numeric(df[res_col], errors='coerce')
                df_plot = df.dropna(subset=[res_col, "ARM", "VISIT"])
                if domain.upper() == "LB":
                    top = df_plot[test_col].value_counts().nlargest(12).index
                    df_plot = df_plot[df_plot[test_col].isin(top)]

                if view_mode == 'mean':
                    fig = px.box(df_plot, x="VISIT", y=res_col, color="ARM", facet_col=test_col, facet_col_wrap=3, color_discrete_sequence=EPHICACY_PALETTE)
                else:
                    fig = px.strip(df_plot, x="VISIT", y=res_col, color="ARM", facet_col=test_col, facet_col_wrap=3, stripmode="group", color_discrete_sequence=EPHICACY_PALETTE)

                fig.update_yaxes(matches=None, showticklabels=True)
                fig.update_layout(template="plotly_white", height=1000)
                fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                figs.append(fig)
                figs.append(px.histogram(df_plot, x=test_col, color="ARM", barmode="group", color_discrete_sequence=EPHICACY_PALETTE))
        
        elif domain.upper() == "DM":
            titles = ["Subject Enrollment Demographics"]
            figs.append(px.histogram(df, x="ARM", color="SEX", barmode="group", color_discrete_sequence=EPHICACY_PALETTE))

        else:
            titles = [f"{domain_label} Profile"]
            cat_col = next((c for c in df.columns if any(k in c for k in ["TERM", "DECOD", "BODSYS"])), df.columns[0])
            figs.append(px.histogram(df, y=cat_col, color="ARM", barmode="group", color_discrete_sequence=EPHICACY_PALETTE))

    except Exception as e: return jsonify({"error": str(e)})
    return jsonify({"plots": [pio.to_json(f) for f in figs], "titles": titles})

if __name__ == "__main__":
    app.run(debug=True, port=5000)