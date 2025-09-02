import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import io


# --- Helper Functions ---

@st.cache_data
def load_data(uploaded_file):
    """Loads a CSV file into a pandas DataFrame, cached for performance."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None


def analyze_plot_and_get_stats(df, x_col, y_col):
    """
    Performs OLS regression, generates a plot, and calculates detailed stats.
    Returns the figure and a dictionary of statistical results.
    """
    # --- 1. Data Preparation ---
    # Create a copy to avoid modifying the original dataframe in the editor
    df_clean = df.copy()
    df_clean[x_col] = pd.to_numeric(df_clean[x_col], errors='coerce')
    df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors='coerce')

    initial_rows = len(df_clean)
    df_clean.dropna(subset=[x_col, y_col], inplace=True)
    final_rows = len(df_clean)

    missing_values = initial_rows - final_rows

    if final_rows < 2:
        st.warning("Not enough valid data points (fewer than 2) to perform regression.")
        return None, None

    X = df_clean[x_col]
    y = df_clean[y_col]
    X_sm = sm.add_constant(X)

    # --- 2. Fit OLS Model using statsmodels ---
    model = sm.OLS(y, X_sm)
    results = model.fit()

    # --- 3. Extract All Statistical Parameters ---
    c, m = results.params['const'], results.params[x_col]
    stderr_c, stderr_m = results.bse['const'], results.bse[x_col]
    conf_int_c = results.conf_int(alpha=0.05).loc['const']
    conf_int_m = results.conf_int(alpha=0.05).loc[x_col]

    stats = {
        "slope": m,
        "slope_stderr": stderr_m,
        "slope_conf_int": tuple(conf_int_m),
        "intercept": c,
        "intercept_stderr": stderr_c,
        "intercept_conf_int": tuple(conf_int_c),
        "x_intercept": -c / m if m != 0 else np.nan,
        "inv_slope": 1 / m if m != 0 else np.nan,
        "r_squared": results.rsquared,
        "syx": np.sqrt(results.mse_resid),
        "f_stat": results.fvalue,
        "p_value": results.f_pvalue,
        "dfn": int(results.df_model),
        "dfd": int(results.df_resid),
        "num_values": int(results.nobs),
        "missing_values": missing_values
    }

    # --- 4. Generate Plot ---
    y_pred = results.predict(X_sm)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots()
    ax.scatter(X, y, color="#0072B2", label="Data Points", alpha=0.8)
    ax.plot(X, y_pred, color="#D55E00", linewidth=2.5, label="Best Fit Line")

    equation = f"Y = {m:.4f}*X + {c:.4f}"
    r2_text = f"R² = {stats['r_squared']:.4f}"

    # Add equation and R² to the plot
    text_str = f"{equation}\n{r2_text}"
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_title(f"Linear Regression of {y_col} vs. {x_col}", fontsize=14)
    ax.legend(loc='best')
    ax.grid(True)
    fig.tight_layout()

    return fig, stats


@st.cache_data
def get_plot_buffers(_fig):
    """Generates in-memory file buffers for downloading the plot."""
    buf_png = io.BytesIO()
    _fig.savefig(buf_png, format="png", dpi=300, bbox_inches="tight")
    return buf_png


# === Streamlit App Interface ===

st.set_page_config(layout="wide", page_title="Advanced Regression Analyzer")

st.title("📊 Advanced Linear Regression Analyzer")
st.write(
    "Interactively generate regression plots and get detailed statistical analysis. Choose to upload a file or edit data directly.")

# --- Sidebar for Controls ---
st.sidebar.header("1. Data Source")
data_source = st.sidebar.radio("Choose your data input method:", ("Enter/Edit Data", "Upload CSV File"),
                               key="data_source_radio")

df = None
if data_source == "Enter/Edit Data":
    st.sidebar.info("Edit the table below. The plot and analysis will update automatically.")
    # Create a sample dataframe for the editor
    sample_data = {
        'Concentration': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Absorbance': [0.15, 0.32, 0.47, 0.61, 0.76, 0.89, 1.05, 1.18, 1.34, 1.51]
    }
    df = pd.DataFrame(sample_data)

    st.subheader("Editable Data Table")
    df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        df = load_data(uploaded_file)

if df is not None and not df.empty:
    st.sidebar.header("2. Column Selection")

    available_columns = df.columns.tolist()
    if len(available_columns) < 2:
        st.error("Your data must have at least two columns.")
    else:
        x_column = st.sidebar.selectbox("X-axis (Independent Var)", available_columns, index=0)
        y_column = st.sidebar.selectbox("Y-axis (Dependent Var)", available_columns,
                                        index=min(1, len(available_columns) - 1))

        if x_column == y_column:
            st.warning("X and Y axes cannot be the same. Please select different columns.")
        else:
            # --- Main Layout ---
            col1, col2 = st.columns((1.2, 1))

            with col1:
                st.subheader("Regression Plot")
                fig, stats = analyze_plot_and_get_stats(df, x_column, y_column)

                if fig and stats:
                    st.pyplot(fig)
                    # Download button
                    buf_png = get_plot_buffers(fig)
                    st.download_button(
                        label="Download Plot as PNG",
                        data=buf_png,
                        file_name=f"regression_{y_column}_vs_{x_column}.png",
                        mime="image/png"
                    )

            with col2:
                st.subheader("Statistical Analysis")
                if stats:
                    st.markdown(f"**Equation:** Y = {stats['slope']:.4f}*X + {stats['intercept']:.4f}")
                    st.markdown(f"**R Square:** {stats['r_squared']:.4f}")

                    with st.expander("Best-fit values", expanded=True):
                        st.text(f"Slope: {stats['slope']:.4f} ± {stats['slope_stderr']:.4f}")
                        st.text(f"Y-intercept: {stats['intercept']:.4f} ± {stats['intercept_stderr']:.4f}")
                        st.text(f"X-intercept: {stats['x_intercept']:.4f}")
                        st.text(f"1/Slope: {stats['inv_slope']:.4f}")

                    with st.expander("95% Confidence Intervals", expanded=True):
                        st.text(f"Slope: {stats['slope_conf_int'][0]:.4f} to {stats['slope_conf_int'][1]:.4f}")
                        st.text(
                            f"Y-intercept: {stats['intercept_conf_int'][0]:.4f} to {stats['intercept_conf_int'][1]:.4f}")

                    with st.expander("Goodness of Fit", expanded=True):
                        st.text(f"R Square: {stats['r_squared']:.4f}")
                        st.text(f"Sy.x (RMSE): {stats['syx']:.4f}")

                    with st.expander("Is slope significantly non-zero?", expanded=True):
                        p_val_text = "< 0.0001" if stats['p_value'] < 0.0001 else f"{stats['p_value']:.4f}"
                        st.text(f"F-statistic: {stats['f_stat']:.2f}")
                        st.text(f"DFn, DFd: {stats['dfn']}, {stats['dfd']}")
                        st.text(f"P Value: {p_val_text}")
                        st.text(
                            f"Deviation from zero?: {'Significant' if stats['p_value'] < 0.05 else 'Not Significant'}")

                    with st.expander("Data Summary", expanded=True):
                        st.text(f"Number of values: {stats['num_values']}")
                        st.text(f"Number of missing values dropped: {stats['missing_values']}")

else:
    st.info("Awaiting data... Please select a data source and provide data to begin analysis.")

