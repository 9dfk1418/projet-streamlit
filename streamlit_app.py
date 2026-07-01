# === streamlit_app.py ===
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(page_title="Road Accidents TP 2024", layout="wide")

PRIMARY = "#1f3a5f"
ACCENT = "#c0392b"
PALETTE = ["#1f3a5f", "#3a6ea5", "#6d9dc5", "#c0392b", "#e67e22", "#95a5a6"]
TEMPLATE = "plotly_white"

DATA_DIR = Path(".")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_raw_data():
    return {
        "caract": pd.read_csv(DATA_DIR / "caract-2024.csv", sep=";", low_memory=False),
        "lieux": pd.read_csv(DATA_DIR / "lieux-2024.csv", sep=";", low_memory=False),
        "usagers": pd.read_csv(DATA_DIR / "usagers-2024.csv", sep=";", low_memory=False),
        "vehicules": pd.read_csv(DATA_DIR / "vehicules-2024.csv", sep=";", low_memory=False),
    }

@st.cache_data
def load_gold_data():
    fact = pd.read_csv(DATA_DIR / "fact_accidents.csv", parse_dates=["date"])
    dim_location = pd.read_csv(DATA_DIR / "dim_location.csv")
    dim_vehicle = pd.read_csv(DATA_DIR / "dim_vehicle.csv")
    dim_user = pd.read_csv(DATA_DIR / "dim_user.csv")
    return fact, dim_location, dim_vehicle, dim_user

raw_data = load_raw_data()
fact, dim_location, dim_vehicle, dim_user = load_gold_data()
fact["month"] = fact["date"].dt.month
fact["weekday"] = fact["date"].dt.day_name()

# ---------------------------------------------------------------------------
# Navigation — follows the assignment structure
# ---------------------------------------------------------------------------
st.sidebar.title("Data Integration TP")
st.sidebar.caption("Road traffic injury accidents - France 2024")

section = st.sidebar.radio(
    "TP Outline",
    [
        "Executive Summary",
        "1.A Dataset Structure",
        "1.B Missing Values",
        "1.C Consistency & Anomalies",
        "1.D Data Quality Summary",
        "2.A Transformations (Silver)",
        "2.B Analytical Model (Gold)",
        "2.C Medallion Architecture",
        "Business Analysis",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Source: data.gouv.fr")

# ---------------------------------------------------------------------------
# EXECUTIVE SUMMARY
# ---------------------------------------------------------------------------
if section == "Executive Summary":
    st.title("Road Traffic Injury Accidents - France 2024")
    st.markdown(
        "Complete profiling, cleaning, and modeling pipeline for road accident data, "
        "structured following a Medallion architecture (Bronze -> Silver -> Gold)."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accidents (caract)", f"{len(raw_data['caract']):,}".replace(",", " "))
    c2.metric("Users involved", f"{len(raw_data['usagers']):,}".replace(",", " "))
    c3.metric("Vehicles involved", f"{len(raw_data['vehicules']):,}".replace(",", " "))
    n_fatal = int((fact["severity_index"] == 2).sum()) if "severity_index" in fact.columns else 0
    c4.metric("Fatal accidents", f"{n_fatal:,}".replace(",", " "))

    st.markdown("### What this dashboard covers")
    st.markdown(
        """
        - **Part 1**: structure of the 4 source tables, completeness, consistency, quality summary
        - **Part 2**: Silver transformations, Gold star schema, Medallion architecture
        - **Business Analysis**: key indicators for road safety
        """
    )

    st.markdown("### Monthly accident trend")
    monthly = fact.groupby("month").size().reindex(range(1, 13), fill_value=0)
    fig = px.line(monthly, markers=True, template=TEMPLATE, color_discrete_sequence=[PRIMARY])
    fig.update_layout(height=350, showlegend=False, xaxis_title="Month", yaxis_title="Accidents")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# 1.A DATASET STRUCTURE
# ---------------------------------------------------------------------------
elif section == "1.A Dataset Structure":
    st.title("A. Dataset Structure")

    table_choice = st.selectbox("Table to inspect", list(raw_data.keys()))
    df = raw_data[table_choice]

    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{len(df):,}".replace(",", " "))
    c2.metric("Columns", df.shape[1])

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Column types**")
        dtype_df = df.dtypes.astype(str).reset_index()
        dtype_df.columns = ["Column", "Type"]
        st.dataframe(dtype_df, use_container_width=True, hide_index=True, height=400)

    with col2:
        st.markdown("**Data preview**")
        st.dataframe(df.head(15), use_container_width=True, height=400)

    data_dict = {
        "caract": {
            "Num_Acc": "Unique accident identifier", "jour/mois/an": "Accident date",
            "hrmn": "Hour and minute", "lum": "Light conditions",
            "dep": "Department", "com": "Municipality", "agg": "Urban area or not",
            "atm": "Atmospheric conditions", "col": "Collision type",
            "lat": "GPS latitude", "long": "GPS longitude",
        },
        "lieux": {
            "catr": "Road category", "circ": "Traffic flow regime",
            "nbv": "Number of lanes", "surf": "Surface condition", "vma": "Speed limit",
        },
        "usagers": {
            "catu": "User category", "grav": "Severity (1=unharmed, 2=killed, 3=hospitalized, 4=minor injury)",
            "sexe": "Sex", "an_nais": "Year of birth",
        },
        "vehicules": {
            "catv": "Vehicle category", "obs": "Fixed obstacle hit",
            "choc": "Impact point", "motor": "Motorization type",
        },
    }
    st.markdown("**Business meaning of main columns**")
    for col, desc in data_dict[table_choice].items():
        st.markdown(f"- `{col}`: {desc}")

# ---------------------------------------------------------------------------
# 1.B MISSING VALUES
# ---------------------------------------------------------------------------
elif section == "1.B Missing Values":
    st.title("B. Missing Values and Completeness")

    st.info(
        "True `NaN` values are rare in this dataset. Unreported fields are often "
        "coded as `-1` instead of being left empty. The chart below combines both."
    )

    sentinel_values = [-1, "-1", "", " "]
    cols = st.columns(2)
    for i, (name, df) in enumerate(raw_data.items()):
        true_na = df.isna().mean() * 100
        sentinel_na = df.isin(sentinel_values).mean() * 100
        total = (true_na + sentinel_na).round(2).sort_values(ascending=False)
        total = total[total > 0]
        with cols[i % 2]:
            st.markdown(f"**{name}**")
            if len(total) == 0:
                st.write("No missing values detected.")
                continue
            fig = px.bar(total, orientation="h", template=TEMPLATE, color_discrete_sequence=[PRIMARY])
            fig.update_layout(height=280, showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                               xaxis_title="% missing", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Criticality and remediation")
    remediation = pd.DataFrame([
        {"Table": "caract", "Column": "lat / long", "Criticality": "Low (0% invalid)", "Action": "No correction needed"},
        {"Table": "lieux", "Column": "vma / nbv", "Criticality": "Medium (~5-6%)", "Action": "Mark as unreported, do not delete"},
        {"Table": "usagers", "Column": "an_nais", "Criticality": "Low (2.06%)", "Action": "Leave age empty if not computable"},
        {"Table": "usagers", "Column": "grav", "Criticality": "Critical if missing", "Action": "Drop the row (target variable, never imputed)"},
    ])
    st.dataframe(remediation, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# 1.C CONSISTENCY & ANOMALIES
# ---------------------------------------------------------------------------
elif section == "1.C Consistency & Anomalies":
    st.title("C. Consistency and Validity Checks")

    tab1, tab2, tab3 = st.tabs(["Value ranges", "Categorical anomalies", "Duplicates"])

    with tab1:
        range_results = pd.DataFrame([
            {"Table": "caract", "Column": "lat / long", "% out of range": 0.00},
            {"Table": "lieux", "Column": "vma", "% out of range": 5.21},
            {"Table": "lieux", "Column": "nbv", "% out of range": 5.97},
            {"Table": "usagers", "Column": "age", "% out of range": 0.00},
        ])
        fig = px.bar(range_results, x="% out of range", y="Column", color="Table",
                     orientation="h", template=TEMPLATE, color_discrete_sequence=PALETTE)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("lieux.vma and lieux.nbv account for the only significant range anomalies.")

    with tab2:
        cat_results = pd.DataFrame([
            {"Table": "caract", "Column": "lum", "Unique values": 5, "Out of nomenclature": 0},
            {"Table": "caract", "Column": "agg", "Unique values": 2, "Out of nomenclature": 0},
            {"Table": "lieux", "Column": "circ", "Unique values": 5, "Out of nomenclature": 0},
            {"Table": "usagers", "Column": "grav", "Unique values": 4, "Out of nomenclature": 0},
            {"Table": "usagers", "Column": "sexe", "Unique values": 3, "Out of nomenclature": 2395},
            {"Table": "vehicules", "Column": "senc", "Unique values": 5, "Out of nomenclature": 0},
        ])
        st.dataframe(cat_results, use_container_width=True, hide_index=True)
        st.warning(
            "usagers.sexe has 3 unique values instead of 2: code -1 (unreported) "
            "is present in addition to 1 (male) and 2 (female)."
        )

    with tab3:
        dup_results = pd.DataFrame([
            {"Table": "caract", "Strict duplicates": 0, "Duplicates on Num_Acc": 0, "Status": "Normal"},
            {"Table": "lieux", "Strict duplicates": 2, "Duplicates on Num_Acc": 15846, "Status": "Not problematic (intersections)"},
            {"Table": "usagers", "Strict duplicates": 0, "Duplicates on Num_Acc": 70785, "Status": "Normal (multiple users/accident)"},
            {"Table": "vehicules", "Strict duplicates": 0, "Duplicates on Num_Acc": 38276, "Status": "Normal (multiple vehicles/accident)"},
        ])
        fig = px.bar(dup_results, x="Table", y="Duplicates on Num_Acc",
                     template=TEMPLATE, color_discrete_sequence=[ACCENT])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dup_results, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# 1.D DATA QUALITY SUMMARY
# ---------------------------------------------------------------------------
elif section == "1.D Data Quality Summary":
    st.title("D. Data Quality Summary")

    quality_summary = pd.DataFrame([
        {"Table": "caract", "Rows": len(raw_data["caract"]), "Main issue": "None major", "Severity": "Low"},
        {"Table": "lieux", "Rows": len(raw_data["lieux"]), "Main issue": "vma / nbv out of range (5-6%)", "Severity": "Medium"},
        {"Table": "usagers", "Rows": len(raw_data["usagers"]), "Main issue": "an_nais missing (2.06%)", "Severity": "Low"},
        {"Table": "vehicules", "Rows": len(raw_data["vehicules"]), "Main issue": "None major", "Severity": "Low"},
    ])

    def color_severity(val):
        colors = {"Low": "background-color: #d4edda", "Medium": "background-color: #fff3cd"}
        return colors.get(val, "")

    st.dataframe(
        quality_summary.style.applymap(color_severity, subset=["Severity"]),
        use_container_width=True, hide_index=True,
    )

    st.markdown("### Impact on downstream analytics")
    st.markdown(
        """
        The four tables are generally clean. GPS coordinates are entirely valid,
        and user age shows no aberrant values once computed.

        The main point of attention is the `lieux` table: the speed limit and
        the number of lanes are out of the expected range for about 5 to 6% of
        rows. Without treatment, these values would have distorted road-type
        statistics in the Gold model.

        The repeated `Num_Acc` values in `lieux`, `usagers`, and `vehicules` are
        not anomalies: they reflect the normal structure of the dataset (multiple
        users, vehicles, or converging lanes per accident).
        """
    )

# ---------------------------------------------------------------------------
# 2.A SILVER TRANSFORMATIONS
# ---------------------------------------------------------------------------
elif section == "2.A Transformations (Silver)":
    st.title("A. Silver Layer Transformations")

    transformation_log = pd.DataFrame([
        {"Table": "caract", "Step": "Standardization", "Detail": "Merged day/month/year into a single date field"},
        {"Table": "caract", "Step": "Enrichment", "Detail": "Added severity_index and time_of_day"},
        {"Table": "lieux", "Step": "Cleaning", "Detail": "vma / nbv out of range marked as unreported"},
        {"Table": "lieux", "Step": "Enrichment", "Detail": "Added catr_label"},
        {"Table": "lieux", "Step": "Deduplication", "Detail": "One row kept per accident (main road segment)"},
        {"Table": "usagers", "Step": "Cleaning", "Detail": "Age computed and bounded to [0,120]"},
        {"Table": "usagers", "Step": "Enrichment", "Detail": "Added age_bucket"},
        {"Table": "vehicules", "Step": "Enrichment", "Detail": "Added is_two_wheeler"},
    ])
    st.dataframe(transformation_log, use_container_width=True, hide_index=True)

    st.markdown("### Effect of deduplication on `lieux`")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows before", f"{len(raw_data['lieux']):,}".replace(",", " "))
    c2.metric("Rows after", f"{len(raw_data['caract']):,}".replace(",", " "))
    c3.metric("Rows removed", f"{len(raw_data['lieux']) - len(raw_data['caract']):,}".replace(",", " "))
    st.caption("After deduplication, lieux has the same row count as caract: one accident, one row.")

# ---------------------------------------------------------------------------
# 2.B GOLD MODEL
# ---------------------------------------------------------------------------
elif section == "2.B Analytical Model (Gold)":
    st.title("B. Analytical Model - Star Schema")

    st.graphviz_chart(f"""
    digraph {{
        rankdir=TB;
        node [shape=box, style=filled, fontname="Helvetica", fontsize=11, fontcolor=white];
        fact [label="fact_accidents\\n(fact table)", color="{ACCENT}"];
        dim_location [label="dim_location", color="{PRIMARY}"];
        dim_vehicle [label="dim_vehicle", color="{PRIMARY}"];
        dim_user [label="dim_user", color="{PRIMARY}"];
        dim_date [label="dim_date", color="{PRIMARY}"];
        dim_location -> fact; dim_vehicle -> fact; dim_user -> fact; dim_date -> fact;
    }}
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("fact_accidents", f"{len(fact):,}".replace(",", " "))
    c2.metric("dim_location", f"{len(dim_location):,}".replace(",", " "))
    c3.metric("dim_vehicle", f"{len(dim_vehicle):,}".replace(",", " "))

    table_choice = st.selectbox("Preview a Gold table", ["fact_accidents", "dim_location", "dim_vehicle", "dim_user"])
    tables = {"fact_accidents": fact, "dim_location": dim_location, "dim_vehicle": dim_vehicle, "dim_user": dim_user}
    st.dataframe(tables[table_choice].head(15), use_container_width=True)

# ---------------------------------------------------------------------------
# 2.C MEDALLION ARCHITECTURE
# ---------------------------------------------------------------------------
elif section == "2.C Medallion Architecture":
    st.title("C. Complete Medallion Architecture")

    st.graphviz_chart(f"""
    digraph {{
        rankdir=LR;
        node [shape=box, style=filled, fontname="Helvetica", fontsize=11, fontcolor=white];
        subgraph cluster_0 {{
            label="Bronze"; style=filled; color="#eef1f5"; fontcolor="{PRIMARY}";
            A1 [label="caract-2024.csv", color="{PRIMARY}"];
            A2 [label="lieux-2024.csv", color="{PRIMARY}"];
            A3 [label="usagers-2024.csv", color="{PRIMARY}"];
            A4 [label="vehicules-2024.csv", color="{PRIMARY}"];
        }}
        subgraph cluster_1 {{
            label="Silver"; style=filled; color="#eef1f5"; fontcolor="{PRIMARY}";
            B1 [label="caract_silver", color="#3a6ea5"];
            B2 [label="lieux_silver", color="#3a6ea5"];
            B3 [label="usagers_silver", color="#3a6ea5"];
            B4 [label="vehicules_silver", color="#3a6ea5"];
        }}
        subgraph cluster_2 {{
            label="Gold"; style=filled; color="#eef1f5"; fontcolor="{PRIMARY}";
            C1 [label="fact_accidents", color="{ACCENT}"];
            C2 [label="dim_location", color="{ACCENT}"];
            C3 [label="dim_vehicle", color="{ACCENT}"];
            C4 [label="dim_user", color="{ACCENT}"];
        }}
        D [label="BI / Streamlit\\nDashboard", shape=box, style=filled, color="#e67e22", fontcolor=white];

        A1 -> B1 -> C1; A2 -> B2 -> C2; A3 -> B3 -> C4; A4 -> B4 -> C3;
        C1 -> D; C2 -> D; C3 -> D; C4 -> D;
    }}
    """)

    st.markdown("### Design justification")
    st.markdown(
        """
        - **Standardization**: dates merged, numeric types enforced to enable calculations.
        - **Targeted cleaning**: out-of-range values (vma, nbv) are marked as unreported
          rather than deleted, to avoid losing other valid columns on the same rows.
        - **Enrichment**: severity_index and time_of_day added, absent from raw data
          but needed for the KPIs.
        - **Targeted deduplication**: only on lieux, to allow a 1-1 join with caract.
        - **Simple star schema**: prioritizes ease of use in a dashboard over full
          granularity (e.g. per-intersection detail available in lieux before dedup).
        """
    )

# ---------------------------------------------------------------------------
# BUSINESS ANALYSIS
# ---------------------------------------------------------------------------
elif section == "Business Analysis":
    st.title("Business Analysis - Road Safety")

    with st.expander("Filter the analysis", expanded=False):
        dep_options = sorted(fact["dep"].dropna().unique().tolist())
        selected_deps = st.multiselect("Department", dep_options)
        month_range = st.slider("Month", 1, 12, (1, 12))

    fact_f = fact.copy()
    if selected_deps:
        fact_f = fact_f[fact_f["dep"].isin(selected_deps)]
    fact_f = fact_f[fact_f["month"].between(month_range[0], month_range[1])]

    n_fatal = int((fact_f["severity_index"] == 2).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accidents", f"{len(fact_f):,}".replace(",", " "))
    c2.metric("Fatal accidents", f"{n_fatal:,}".replace(",", " "))
    c3.metric("Fatality rate", f"{(n_fatal/len(fact_f)*100 if len(fact_f) else 0):.1f} %")
    c4.metric("Departments", fact_f["dep"].nunique())

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Severity breakdown")
        grav_labels = {1: "Unharmed", 2: "Killed", 3: "Hospitalized", 4: "Minor injury"}
        grav_counts = fact_f["severity_index"].map(grav_labels).value_counts()
        fig = go.Figure(data=[go.Pie(labels=grav_counts.index, values=grav_counts.values, hole=0.5,
                                      marker=dict(colors=PALETTE))])
        fig.update_layout(height=340, template=TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### By time of day")
        if "time_of_day" in fact_f.columns:
            counts = fact_f["time_of_day"].value_counts()
            fig = px.bar(counts, template=TEMPLATE, color_discrete_sequence=[PRIMARY])
            fig.update_layout(height=340, showlegend=False, xaxis_title="", yaxis_title="Accidents")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Accident map")
    map_data = fact_f[["lat", "long"]].dropna().rename(columns={"long": "lon"})
    if len(map_data) > 5000:
        map_data = map_data.sample(5000, random_state=42)
        st.caption("Sample of 5,000 points shown for performance.")
    st.map(map_data, size=8)

    st.markdown("### Monthly severity breakdown (static)")
    fig, ax = plt.subplots(figsize=(10, 4))
    cross = pd.crosstab(fact_f["month"], fact_f["severity_index"])
    cross.plot(kind="bar", stacked=True, ax=ax, color=PALETTE)
    ax.set_xlabel("Month"); ax.set_ylabel("Accidents")
    ax.legend(title="Severity", bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.despine()
    st.pyplot(fig)