from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from flask.json.provider import DefaultJSONProvider
import pandas as pd
import pyreadstat
import plotly.express as px
import plotly.io as pio
import os
import numpy as np
from functools import lru_cache

class CustomJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        from datetime import date, datetime
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)

app = Flask(__name__)
app.secret_key = "pharmasug_2026_final_pro"
app.json = CustomJSONProvider(app)

DATA_PATH = "./SDTM_ADAM_SAS/"

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

@lru_cache(maxsize=32)
def load_data(file_name):
    path = os.path.join(DATA_PATH, f"{file_name.lower()}.sas7bdat")
    if not os.path.exists(path): return None
    try:
        df, _ = pyreadstat.read_sas7bdat(path)
        for col in df.select_dtypes(['object']).columns:
            df[col] = df[col].astype(str).str.strip()
        
        if "ARM" not in df.columns:
            dm_path = os.path.join(DATA_PATH, "dm.sas7bdat")
            if os.path.exists(dm_path):
                dm_df, _ = pyreadstat.read_sas7bdat(dm_path)
                dm_df["USUBJID"] = dm_df["USUBJID"].astype(str).str.strip()
                df = pd.merge(df, dm_df[["USUBJID", "ARM", "SEX"]], on="USUBJID", how="left")
        return df
    except: return None

def get_clinical_summary(df, group_cols, y_col):
    try:
        stats = df.groupby(group_cols)[y_col].agg(['count', 'mean', 'std', 'median', 'min', 'max']).round(2)
        stats['Mean (SD)'] = stats['mean'].astype(str) + " (" + stats['std'].fillna(0).astype(str) + ")"
        stats['Range'] = stats['min'].astype(str) + " - " + stats['max'].astype(str)
        stats = stats[['count', 'Mean (SD)', 'median', 'Range']].reset_index()
        stats.rename(columns={'count': 'N', 'median': 'Median'}, inplace=True)
        return stats.to_html(classes='summary-table', index=False)
    except: return ""

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    if data.get("username") == "Python_Ephicacy" and data.get("password") == "admin":
        session["logged_in"] = True
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "Access Denied: Please verify credentials."})

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route("/")
def index():
    show_login = not session.get("logged_in")
    return render_template("dashboard.html", domains=DOMAINS, show_login=show_login)

@app.route("/get_metadata/<dataset>")
def get_metadata(dataset):
    if not session.get("logged_in"): return jsonify({"error": "Auth required"})
    df = load_data(dataset)
    if df is None: return jsonify({"error": "No file"})
    return jsonify({"columns": list(df.columns), "numeric_cols": list(df.select_dtypes(include=[np.number]).columns)})

@app.route("/get_analysis/<dataset>")
def get_analysis(dataset):
    if not session.get("logged_in"): return jsonify({"error": "Auth required"})
    
    x, y, grp = request.args.get('x'), request.args.get('y'), request.args.get('grp')
    chart_type = request.args.get('type')
    
    df = load_data(dataset)
    if df is None: return jsonify({"error": "Data error"})
    
    # Force Grouping to None if Pie Chart is selected
    if chart_type == "pie":
        grp = None
    else:
        grp = None if grp == "NONE" else grp
    
    param_col = next((c for c in [f"{dataset.upper()}TESTCD", "PARAMCD"] if c in df.columns), None)

    if chart_type in ["box", "line"] and y and y in df.columns:
        df[y] = pd.to_numeric(df[y], errors='coerce')
        plot_df = df.dropna(subset=[y]).copy()

        if param_col:
            top_p = plot_df[param_col].value_counts().nlargest(6).index
            plot_df = plot_df[plot_df[param_col].isin(top_p)]

        if chart_type == "box":
            main_title = f"<b>Distribution Analysis</b>: {y} across {x}"
            fig = px.box(plot_df, x=x, y=y, color=grp, facet_col=param_col, facet_col_wrap=3, title=main_title)
        else: # Line
            main_title = f"<b>Temporal Analysis</b>: Mean {y} over Clinical Visits"
            s_col = next((c for c in ["AVISITN", "VISITNUM", "ADY"] if c in plot_df.columns), None)
            l_df = plot_df.groupby([c for c in [s_col, x, grp, param_col] if c])[y].mean().reset_index()
            if s_col: l_df = l_df.sort_values(s_col)
            fig = px.line(l_df, x=x, y=y, color=grp, facet_col=param_col, facet_col_wrap=3, markers=True, title=main_title)

        fig.update_yaxes(matches=None)
        fig.for_each_annotation(lambda a: a.update(text=f"<span style='color:#004a99; font-weight:bold;'>{a.text.split('=')[-1]}</span>"))
        
        fig.update_layout(
            template="plotly_white",
            height=750 if param_col else 550,
            title_font=dict(size=22, color='#004a99'),
            legend=dict(font=dict(size=15), borderwidth=1, bordercolor="#e2e8f0", bgcolor="rgba(255,255,255,0.8)", orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            margin=dict(r=160, t=100, l=60, b=60) 
        )
        summary_html = get_clinical_summary(plot_df, [c for c in [param_col, grp, x] if c], y)
        return jsonify({"plots": [pio.to_json(fig)], "table": summary_html})
    else:
        # Bar/Pie logic
        t_col = x if x else (param_col if param_col else df.columns[0])
        counts = df.groupby([grp, t_col]).size().reset_index(name='Count') if grp else df.groupby([t_col]).size().reset_index(name='Count')
        
        if chart_type == "pie":
            title = f"<b>Proportional Composition</b>: {t_col}"
            fig = px.pie(df, names=t_col, title=title)
        else:
            title = f"<b>Incidence Analysis</b>: {t_col} Frequency"
            fig = px.bar(counts.head(20), x=t_col, y="Count", color=grp, barmode="group", title=title)
        
        fig.update_layout(
            template="plotly_white", 
            title_font=dict(size=22, color='#004a99'),
            legend=dict(font=dict(size=15), borderwidth=1, bordercolor="#e2e8f0"),
            margin=dict(r=160, t=100, l=60, b=60)
        )
        return jsonify({"plots": [pio.to_json(fig)], "table": counts.to_html(classes='summary-table', index=False)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)