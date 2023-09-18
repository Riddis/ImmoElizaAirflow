import streamlit


def build_path():
    """Builds path to csv locations"""
    cwd = Path.cwd()
    csv_cleaned_path = 'dags/data/dataframe_cleaned_visual.csv'
    src_path = (cwd / csv_cleaned_path).resolve()

    return src_path

def get_csv(src_path):
    """Parse the csv located at 'data/dataframe.csv'"""
    csv = pd.read_csv(src_path, index_col=0)

    return csv

def make_plots(df):
    #Do stuff
    

def stream():
    src_path = build_path()
    df = get_csv(src_path)
    plot1, plot2, plot3 = make_plots(df)