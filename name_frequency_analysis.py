#!/usr/bin/env uv python
# /// script
# dependencies = [
#   "pandas",
#   "requests",
#   "textual",
#   "textual-plotext",
#   "python-levenshtein",
#   "jellyfish"
# ]
# ///

# baby_name_analyzer_tui.py
# A Textual User Interface (TUI) application to analyze and visualize U.S. baby name data.

import pandas as pd
from pathlib import Path
import zipfile
import requests
import io
from typing import Optional, List, Tuple, Dict, Any
import math
import json
from functools import reduce

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll, Grid, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Header, Footer, Button, Input, DataTable, Static, LoadingIndicator, RadioSet, RadioButton, TabbedContent, TabPane, Checkbox, Markdown
from textual.validation import Integer
from textual_plotext import PlotextPlot
import Levenshtein
import jellyfish

# --- Configuration ---
DATA_URL = "https://www.ssa.gov/oact/babynames/names.zip"
DATA_DIR = Path("baby_names_data")
STATE_FILE = Path("app_state.json")
START_YEAR = 1880
END_YEAR = 2023

# --- Help Screen ---

HELP_TEXT = """
## Name Frequency Analysis

This application analyzes U.S. baby name data from the Social Security Administration (SSA) for the years 1880-2023.

### How to Use

- Use the **tabs** on the left to select an analysis type.
- Use the **radio buttons** and **checkbox** to filter the data. The view will update automatically.
- Click on a **name** in any results table to quickly get details about that name.
- Click on a **year** in a details table to fetch the top names for that year.

### Analysis Tabs Explained

* **Single Name:** View the historical popularity plot, get a statistical summary (including the "Name Signature" for its peak year), or find similarly spelled or sounding names.
* **Top Names:** See a ranked list of the most popular names for a specific year. Use the Previous/Next buttons to page through years.
* **Analysis:**
    * *Plot Unique Names*: See how vocabulary has grown over the years.
    * *Biggest Movers*: Find names that had the biggest jump or fall in popularity year-over-year.
    * *Enduring Popularity*: Discover names that have been consistently popular over the entire period.
    * *Name Origins*: See how many new names were introduced each decade.
* **Compare:** Plot multiple names on the same graph to compare their histories.

### Data Source

Data is provided by the [U.S. Social Security Administration](https://www.ssa.gov/oact/babynames/limits.html).

**Press ESC, Q, or ? to close this screen.**
"""

class HelpScreen(ModalScreen):
    """A modal screen that displays help information."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Close Help"),
        ("q", "app.pop_screen", "Close Help"),
        ("?", "app.pop_screen", "Close Help"),
    ]

    def compose(self) -> ComposeResult:
        with Grid(id="help-grid"):
            yield Markdown(HELP_TEXT)

# --- Data Processing Class ---

class DataAnalyzer:
    """Handles all data fetching, loading, and processing logic."""
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.df: Optional[pd.DataFrame] = None
        self.yearly_totals: Optional[pd.DataFrame] = None
        self.all_time_totals: Optional[pd.DataFrame] = None
        self.unique_name_sex_pairs: Optional[pd.DataFrame] = None
        self.first_year_df: Optional[pd.DataFrame] = None
        self.phonetic_df: Optional[pd.DataFrame] = None

    def download_and_extract_data(self):
        if self.data_dir.exists():
            yield "Data directory already exists. Skipping download."
            return

        yield "Downloading data..."
        try:
            response = requests.get(DATA_URL, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                for member in z.infolist():
                    if member.is_dir() or not member.filename.startswith('yob'):
                        continue
                    target_path = self.data_dir / Path(member.filename).name
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(target_path, "wb") as f:
                        f.write(z.read(member.filename))
            yield "Data download complete."
        except requests.exceptions.RequestException as e:
            yield f"Network error downloading data: {e}"
        except (zipfile.BadZipFile, IOError) as e:
            yield f"Error processing data file: {e}"

    def load_data(self) -> str:
        if self.df is not None:
            return "Data already loaded."

        all_years_data = []
        error_count = 0
        for year in range(START_YEAR, END_YEAR + 1):
            file_path = self.data_dir / f"yob{year}.txt"
            if file_path.exists():
                try:
                    year_df = pd.read_csv(file_path, names=["Name", "Sex", "Count"])
                    year_df["Year"] = year
                    all_years_data.append(year_df)
                except pd.errors.ParserError:
                    error_count += 1

        if not all_years_data:
            self.df = None
            return "No data files found."

        self.df = pd.concat(all_years_data, ignore_index=True)
        self.yearly_totals = self.df.groupby("Year")['Count'].sum().reset_index().rename(columns={'Count': 'Total'})
        self.all_time_totals = self.df.groupby(['Name', 'Sex'])['Count'].sum().reset_index()
        self.unique_name_sex_pairs = self.df[['Name', 'Sex']].drop_duplicates().reset_index(drop=True)
        self.first_year_df = self.df.groupby(['Name', 'Sex'])['Year'].min().reset_index().rename(columns={'Year': 'FirstYear'})
        
        self.phonetic_df = self.unique_name_sex_pairs.copy()
        self.phonetic_df['Metaphone'] = self.phonetic_df['Name'].apply(jellyfish.metaphone)

        status = f"Data loaded with {len(self.df):,} records."
        if error_count > 0:
            status += f" [bold red]({error_count} file(s) failed to parse)[/]"
        return status
        
    def get_name_origins_by_decade(self, sex_filter: str) -> Optional[pd.DataFrame]:
        """Calculates the number of new unique names introduced per decade."""
        if self.first_year_df is None:
            return None
        
        source_df = self.first_year_df.copy()
        if sex_filter == "Male":
            source_df = source_df[source_df['Sex'] == 'M']
        elif sex_filter == "Female":
            source_df = source_df[source_df['Sex'] == 'F']

        # Get unique names and their first year
        unique_first_years = source_df.groupby('Name')['FirstYear'].min().reset_index()
        
        unique_first_years['Decade'] = (unique_first_years['FirstYear'] // 10) * 10
        decade_counts = unique_first_years.groupby('Decade')['Name'].nunique().reset_index()
        decade_counts = decade_counts.rename(columns={'Name': 'NewNameCount'})
        return decade_counts

    def get_phonetically_similar_names(self, target_name: str, sex_filter: str, limit: int = 20) -> Tuple[Optional[pd.DataFrame], int, Optional[int]]:
        if self.phonetic_df is None or self.all_time_totals is None or self.first_year_df is None:
            return None, 0, None

        target_name_lower = target_name.lower()
        target_metaphone = jellyfish.metaphone(target_name)
        if not target_metaphone:
            return None, 0, None

        target_name_totals = self.all_time_totals[self.all_time_totals.Name.str.lower() == target_name_lower]
        target_name_first_year = self.first_year_df[self.first_year_df.Name.str.lower() == target_name_lower]
        
        if sex_filter == "Male":
            target_name_totals = target_name_totals[target_name_totals.Sex == 'M']
            target_name_first_year = target_name_first_year[target_name_first_year.Sex == 'M']
        elif sex_filter == "Female":
            target_name_totals = target_name_totals[target_name_totals.Sex == 'F']
            target_name_first_year = target_name_first_year[target_name_first_year.Sex == 'F']
            
        target_name_count = target_name_totals['Count'].sum()
        original_first_year = int(target_name_first_year['FirstYear'].min()) if not target_name_first_year.empty else None

        similar_names_df = self.phonetic_df[
            (self.phonetic_df['Metaphone'] == target_metaphone) &
            (self.phonetic_df['Name'].str.lower() != target_name_lower)
        ].copy()

        if sex_filter == "Male":
            similar_names_df = similar_names_df[similar_names_df['Sex'] == 'M']
        elif sex_filter == "Female":
            similar_names_df = similar_names_df[similar_names_df['Sex'] == 'F']

        results = pd.merge(similar_names_df, self.all_time_totals, on=['Name', 'Sex'], how='left')
        results = pd.merge(results, self.first_year_df, on=['Name', 'Sex'], how='left')

        return results.sort_values(by='Count', ascending=False).head(limit), target_name_count, original_first_year

    def get_similar_names(self, target_name: str, sex_filter: str, limit: int = 20, max_distance: int = 2) -> Tuple[Optional[pd.DataFrame], int, Optional[int]]:
        if self.unique_name_sex_pairs is None or self.all_time_totals is None or self.first_year_df is None:
            return None, 0, None

        target_name_lower = target_name.lower()
        target_name_totals = self.all_time_totals[self.all_time_totals.Name.str.lower() == target_name_lower]
        target_name_first_year = self.first_year_df[self.first_year_df.Name.str.lower() == target_name_lower]
        
        if sex_filter == "Male":
            target_name_totals = target_name_totals[target_name_totals.Sex == 'M']
            target_name_first_year = target_name_first_year[target_name_first_year.Sex == 'M']
        elif sex_filter == "Female":
            target_name_totals = target_name_totals[target_name_totals.Sex == 'F']
            target_name_first_year = target_name_first_year[target_name_first_year.Sex == 'F']
        
        target_name_count = target_name_totals['Count'].sum()
        original_first_year = int(target_name_first_year['FirstYear'].min()) if not target_name_first_year.empty else None

        source_df = self.unique_name_sex_pairs.copy()
        if sex_filter == "Male":
            source_df = source_df[source_df['Sex'] == 'M']
        elif sex_filter == "Female":
            source_df = source_df[source_df['Sex'] == 'F']

        source_df['Distance'] = source_df['Name'].apply(
            lambda name: Levenshtein.distance(target_name_lower, name.lower())
        )
        
        similar_names_df = source_df[
            (source_df['Distance'] > 0) & (source_df['Distance'] <= max_distance)
        ]

        results = pd.merge(similar_names_df, self.all_time_totals, on=['Name', 'Sex'], how='left')
        results = pd.merge(results, self.first_year_df, on=['Name', 'Sex'], how='left')

        if target_name_count > 0:
            results['PopularityRatio'] = results['Count'] / target_name_count
        else:
            results['PopularityRatio'] = float('nan')

        return results.sort_values(by=['Distance', 'Count'], ascending=[True, False]).head(limit), target_name_count, original_first_year

    def get_name_history(self, name: str, sex_filter: str) -> Optional[pd.DataFrame]:
        if self.df is None or self.yearly_totals is None: return None

        name_data_raw = self.df[self.df["Name"].str.lower() == name.lower()].copy()
        if sex_filter == "Male":
            name_data_raw = name_data_raw[name_data_raw.Sex == "M"]
        elif sex_filter == "Female":
            name_data_raw = name_data_raw[name_data_raw.Sex == "F"]

        all_years = pd.DataFrame({'Year': range(START_YEAR, END_YEAR + 1)})

        if name_data_raw.empty:
            return all_years.assign(Name=name, Sex=sex_filter, Count=0, Total=0, Percentage=0)

        sexes = name_data_raw['Sex'].unique()

        all_combinations = pd.MultiIndex.from_product(
            [all_years['Year'], name_data_raw['Name'].unique(), sexes],
            names=['Year', 'Name', 'Sex']
        ).to_frame(index=False)

        name_data = pd.merge(all_combinations, name_data_raw, on=['Year', 'Name', 'Sex'], how='left').fillna(0)
        name_data = pd.merge(name_data, self.yearly_totals, on="Year", how="left").fillna(0)
        
        name_data["Percentage"] = (name_data["Count"] / name_data["Total"].replace(0, 1)) * 100
        return name_data

    def get_name_details(self, name: str, sex_filter: str) -> Optional[Dict[str, Any]]:
        if self.df is None or self.all_time_totals is None: return None

        name_df = self.df[self.df["Name"].str.lower() == name.lower()].copy()
        if sex_filter == "Male": name_df = name_df[name_df.Sex == "M"]
        elif sex_filter == "Female": name_df = name_df[name_df.Sex == "F"]

        if name_df.empty: return None

        total_births = name_df['Count'].sum()
        first_appearance = name_df['Year'].min()
        peak_year_row = name_df.loc[name_df['Count'].idxmax()]
        peak_year = int(peak_year_row['Year'])
        peak_year_count = peak_year_row['Count']

        name_df['Decade'] = (name_df['Year'] // 10) * 10
        decade_popularity = name_df.groupby('Decade')['Count'].sum()
        most_popular_decade = decade_popularity.idxmax() if not decade_popularity.empty else "N/A"

        name_sex_totals = name_df.groupby('Sex')['Count'].sum()
        ranks = {}
        for sex, _ in name_sex_totals.items():
            sex_specific_totals = self.all_time_totals[self.all_time_totals.Sex == sex].sort_values(by="Count", ascending=False).reset_index(drop=True)
            rank_series = sex_specific_totals[sex_specific_totals.Name.str.lower() == name.lower()].index
            if not rank_series.empty:
                ranks[sex] = rank_series[0] + 1
        
        signature_names = self.get_top_names(year=peak_year, n=5, sex_filter="Both")
        top_males_peak = signature_names.get('male', pd.DataFrame())
        top_females_peak = signature_names.get('female', pd.DataFrame())

        return {
            "Total Births": total_births,
            "First Appearance": first_appearance,
            "Peak Year": peak_year,
            "Peak Year Count": peak_year_count,
            "Most Popular Decade": f"{most_popular_decade}s",
            "All-Time Rank": ranks,
            "Peak Year Signature (Male)": top_males_peak['Name'].tolist() if not top_males_peak.empty else [],
            "Peak Year Signature (Female)": top_females_peak['Name'].tolist() if not top_females_peak.empty else []
        }

    def get_unique_names_per_year(self, sex_filter: str) -> Optional[pd.DataFrame]:
        if self.df is None: return None
        source_df = self.df
        if sex_filter == "Male": source_df = self.df[self.df.Sex == "M"]
        elif sex_filter == "Female": source_df = self.df[self.df.Sex == "F"]
        return source_df.groupby('Year')['Name'].nunique().reset_index()

    def get_biggest_movers(self, n: int, sex_filter: str, normalized: bool) -> dict:
        if self.df is None: return {}
        source_df = self.df.copy()
        if sex_filter != "Both":
            source_df = source_df[source_df.Sex == ("M" if sex_filter == "Male" else "F")]
        source_df = source_df.sort_values(by=["Name", "Sex", "Year"])
        if normalized:
            source_df['Prev_Count'] = source_df.groupby(['Name', 'Sex'])['Count'].shift(1).fillna(0)
            change_df = source_df[source_df.Prev_Count > 0].copy()
            change_df['Change'] = 100 * (change_df['Count'] - change_df['Prev_Count']) / change_df['Prev_Count']
        else:
            change_df = source_df.copy()
            change_df['Change'] = change_df.groupby(['Name', 'Sex'])['Count'].diff().fillna(0)
        top_gainers = change_df.nlargest(n, 'Change').reset_index(drop=True)
        top_losers = change_df.nsmallest(n, 'Change').reset_index(drop=True)
        return {"gainers": top_gainers, "losers": top_losers}

    def get_enduring_popularity(self, n: int, sex_filter: str) -> Optional[pd.DataFrame]:
        if self.df is None or self.yearly_totals is None: return None
        source_df = self.df
        if sex_filter != "Both":
            source_df = source_df[source_df.Sex == ("M" if sex_filter == "Male" else "F")]
        all_names = source_df[['Name', 'Sex']].drop_duplicates()
        all_years = pd.DataFrame({'Year': range(START_YEAR, END_YEAR + 1)})
        all_combinations = all_names.merge(all_years, how='cross')
        df_full = pd.merge(all_combinations, source_df, on=['Name', 'Sex', 'Year'], how='left').fillna(0)
        df_full = pd.merge(df_full, self.yearly_totals, on='Year')
        df_full["Percentage"] = (df_full["Count"] / df_full["Total"]) * 100
        enduring_popularity = df_full.groupby(['Name', 'Sex'])['Percentage'].mean().reset_index()
        return enduring_popularity.sort_values(by="Percentage", ascending=False).head(n)

    def get_top_names(self, year: int, n: int, sex_filter: str) -> dict:
        if self.df is None or self.yearly_totals is None: return {}
        if not (START_YEAR <= year <= END_YEAR): return {}
        year_data = self.df[self.df['Year'] == year].copy()
        if year_data.empty: return {}

        prev_year_data = self.df[self.df['Year'] == year - 1].copy() if year > START_YEAR else pd.DataFrame()
        
        total_for_year_series = self.yearly_totals.loc[self.yearly_totals.Year == year, 'Total']
        if total_for_year_series.empty: return {}
        total_for_year = total_for_year_series.iloc[0]
        year_data["Percentage"] = (year_data["Count"] / total_for_year) * 100
        
        results: Dict[str, pd.DataFrame] = {}

        if sex_filter in ("Both", "Female"):
            top_females = year_data[year_data.Sex == 'F'].sort_values(by="Count", ascending=False).head(n).reset_index()
            if not prev_year_data.empty:
                prev_ranks_f = {name: i + 1 for i, name in prev_year_data[prev_year_data.Sex == 'F'].sort_values('Count', ascending=False)['Name'].reset_index(drop=True).items()}
                top_females['Mvmt'] = top_females.apply(lambda row: prev_ranks_f.get(row['Name'], 0) - (row.name + 1) if prev_ranks_f.get(row['Name']) else 'New', axis=1)
            else:
                top_females['Mvmt'] = 'New'
            results['female'] = top_females

        if sex_filter in ("Both", "Male"):
            top_males = year_data[year_data.Sex == 'M'].sort_values(by="Count", ascending=False).head(n).reset_index()
            if not prev_year_data.empty:
                prev_ranks_m = {name: i + 1 for i, name in prev_year_data[prev_year_data.Sex == 'M'].sort_values('Count', ascending=False)['Name'].reset_index(drop=True).items()}
                top_males['Mvmt'] = top_males.apply(lambda row: prev_ranks_m.get(row['Name'], 0) - (row.name + 1) if prev_ranks_m.get(row['Name']) else 'New', axis=1)
            else:
                top_males['Mvmt'] = 'New'
            results['male'] = top_males
        return results

# --- Textual Application ---

class NameAnalysisApp(App):
    """A Textual app to analyze baby names."""

    TITLE = "Name Frequency Analysis"

    CSS = """
    #app-grid {
        layout: grid;
        grid-size: 2;
        grid-columns: 45 1fr;
        grid-rows: 1fr;
        height: 100%;
        width: 100%;
    }
    #controls-pane {
        layout: grid;
        grid-rows: 1fr auto;
        padding: 1 2;
        border-right: solid $accent;
    }
    #results-pane {
        padding: 0 1;
        height: 100%;
        layout: grid;
        grid-rows: 1fr;
        grid-columns: 1fr;
    }
    .header {
        background: $primary-background-darken-1;
        color: $text;
        padding: 0 1;
        margin-top: 1;
        text-style: bold;
    }
    Button {
        width: 100%;
        margin-top: 1;
    }
    Input {
        margin-top: 1;
    }
    Input.-invalid {
        border: heavy red;
    }
    RadioSet, Checkbox {
        margin-top: 1;
    }
    #control-tabs {
        height: auto;
    }
    TabPane {
        padding-top: 1;
    }
    #year-nav-buttons {
        layout: horizontal;
        height: auto;
        width: 100%;
        align: center middle;
        margin-top: 1;
    }
    #year-nav-buttons Button {
        width: 1fr;
    }
    #plot-container {
        height: 100%;
    }
    #plot_view {
        height: 60%;
    }
    #top-names-table {
        height: 100%;
    }
    #plot-details-table {
        height: 40%;
    }
    #status-container {
        border: round $panel-lighten-1;
        border-title-color: $accent;
        padding: 0 1;
        margin-top: 1;
        height: 5;
    }
    #status_widget {
        height: 100%;
    }
    .hidden {
        display: none;
    }
    .separator-column {
        text-align: center;
        color: $text-muted;
    }
    .input-label {
        margin-top: 1;
    }
    #comparison-inputs {
        height: auto;
    }
    #loading-overlay {
        background: $surface 50%;
        width: 100%;
        height: 100%;
        align: center middle;
        display: none;
    }
    HelpScreen {
        align: center middle;
    }
    #help-grid {
        grid-size: 1;
        grid-gutter: 1 2;
        padding: 0 1;
        width: 80;
        height: 22;
        border: thick $primary;
        background: $surface;
    }
    """

    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("?", "show_help", "Show Help"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.data_analyzer = DataAnalyzer(DATA_DIR)
        self.status_widget = Static("Loading...")
        self.loaded_state: Optional[dict] = None
        self.current_view: str = "plot"

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="app-grid"):
            with Container(id="controls-pane"):
                with VerticalScroll():
                    with TabbedContent(id="control-tabs"):
                        with TabPane("Single Name", id="single-name-controls"):
                            yield Static("Name:", classes="input-label")
                            yield Input(placeholder="Enter a name...", id="name_input")
                            yield Button("Plot History", variant="primary", id="plot_button", disabled=True)
                            yield Button("Get Details", variant="success", id="get_details_button", disabled=True)
                            yield Button("Similar by Spelling", variant="warning", id="find_similar_button", disabled=True)
                            yield Button("Similar by Sound", variant="warning", id="find_phonetic_button", disabled=True)
                        with TabPane("Top Names", id="top-names-controls"):
                            yield Static("Year:", classes="input-label")
                            yield Input(
                                placeholder=f"Year ({START_YEAR}-{END_YEAR})", 
                                id="year_input",
                                validators=[Integer(minimum=START_YEAR, maximum=END_YEAR)],
                                value=str(END_YEAR-1)
                            )
                            yield Static("Number of Results:", classes="input-label")
                            yield Input(
                                placeholder="How many? (e.g., 10)", 
                                value="10", 
                                id="n_input",
                                validators=[Integer(minimum=1, maximum=100)]
                            )
                            with Horizontal(id="year-nav-buttons"):
                                yield Button("Previous Year", id="prev_year_button", disabled=True)
                                yield Button("Next Year", id="next_year_button", disabled=True)
                            yield Button("Get Top Names", variant="primary", id="top_names_button", disabled=True)
                        with TabPane("Analysis", id="analysis-controls"):
                            yield Button("Plot Unique Names per Year", variant="primary", id="unique_names_button", disabled=True)
                            yield Button("Biggest Movers (Y-o-Y)", variant="primary", id="biggest_movers_button", disabled=True)
                            yield Button("Enduring Popularity", variant="primary", id="enduring_popularity_button", disabled=True)
                            yield Button("Name Origins by Decade", variant="primary", id="name_origins_button", disabled=True)
                        with TabPane("Compare", id="compare-controls"):
                            with VerticalScroll(id="comparison-inputs"):
                                yield Input(placeholder="Name 1", classes="comparison-input")
                                yield Input(placeholder="Name 2", classes="comparison-input")
                            yield Button("Add Name Field", id="add_name_field_button")
                            yield Button("Remove Last Name", id="remove_name_field_button", disabled=True)
                            yield Button("Plot Comparison", variant="primary", id="plot_comparison_button", disabled=True)
                with Container():
                    yield Static("Options", classes="header")
                    with RadioSet(id="sex_filter"):
                        yield RadioButton("Both", value=True, id="sex_both")
                        yield RadioButton("Male", id="sex_male")
                        yield RadioButton("Female", id="sex_female")
                    yield Checkbox("Normalize Data (for Plots)", id="normalize_data")
                    with Container(id="status-container"):
                        yield self.status_widget
            with Container(id="results-pane"):
                with VerticalScroll(id="plot-container"):
                    yield PlotextPlot(id="plot_view")
                    yield DataTable(id="plot-details-table", classes="hidden")
                yield DataTable(id="top-names-table", classes="hidden")
                with Container(id="loading-overlay"):
                    yield LoadingIndicator()
        yield Footer()

    def action_show_help(self) -> None:
        """Show the help screen."""
        self.push_screen(HelpScreen())

    def on_mount(self) -> None:
        self._load_state()
        self.query_one("#status-container").border_title = "Status"
        self.status_widget.update("Checking for data...")
        self.load_data_worker()
        self._draw_welcome_plot()
        self._update_compare_buttons()
        self._update_year_nav_buttons()

    @work(exclusive=True, thread=True)
    def load_data_worker(self) -> None:
        self.call_from_thread(self.query_one, "#loading-overlay").display = True
        for status in self.data_analyzer.download_and_extract_data():
            self.call_from_thread(self.status_widget.update, status)
        status = self.data_analyzer.load_data()
        self.call_from_thread(self.status_widget.update, status)
        self.call_from_thread(self.query_one, "#loading-overlay").display = False
        if self.data_analyzer.df is not None:
            self.call_from_thread(self.status_widget.update, f"[bold green]Data loaded! Ready to analyze.[/]\n({len(self.data_analyzer.df):,} records)")
            for button in self.query(Button):
                if button.id != "remove_name_field_button":
                    button.disabled = False
            self._update_compare_buttons()
            self._update_year_nav_buttons()
        if self.loaded_state:
            self.call_later(self._restore_last_view)
            
    def _refresh_current_view(self):
        """Re-runs the analysis for the currently active view."""
        if self.data_analyzer.df is None:
            return

        actions = {
            "plot": self._draw_plot,
            "details": self._get_name_details_with_worker,
            "similar": self._find_similar_names_with_worker,
            "phonetic": self._find_phonetic_names_with_worker,
            "table": self._populate_table_with_worker,
            "unique": self._draw_unique_names_plot,
            "movers": self._draw_biggest_movers_plot_with_worker,
            "compare": self._draw_comparison_plot,
            "enduring": self._draw_enduring_popularity_with_worker,
            "origins": self._draw_name_origins_with_worker
        }
        
        if self.current_view in actions:
            is_plot_view = self.current_view in ("plot", "unique", "movers", "compare", "origins")
            self.query_one("#plot-container").display = is_plot_view
            self.query_one("#top-names-table").display = not is_plot_view
            actions[self.current_view]()


    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Called when the sex filter is changed."""
        self._refresh_current_view()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Called when the normalize checkbox is changed."""
        self._refresh_current_view()

    def _restore_last_view(self) -> None:
        last_view = self.loaded_state.get("active_view")
        if not last_view: return
        self.current_view = last_view
        self._refresh_current_view()

    def on_app_quit(self) -> None:
        try:
            self._save_state()
        except Exception:
            pass

    def _save_state(self) -> None:
        try:
            pressed_button = self.query_one(RadioSet).pressed_button
            sex_filter_id = pressed_button.id if pressed_button else None
            state = {
                "name_input": self.query_one("#name_input", Input).value,
                "year_input": self.query_one("#year_input", Input).value,
                "n_input": self.query_one("#n_input", Input).value,
                "sex_filter": sex_filter_id,
                "normalize_data": self.query_one("#normalize_data", Checkbox).value,
                "control_tab": self.query_one("#control-tabs").active,
                "active_view": self.current_view,
                "compare_names": [i.value for i in self.query(".comparison-input")]
            }
            STATE_FILE.write_text(json.dumps(state, indent=4))
        except Exception as e:
            self.status_widget.update(f"[bold red]Error saving state: {e}[/]")

    def _load_state(self) -> None:
        if not STATE_FILE.exists(): return
        try:
            self.loaded_state = json.loads(STATE_FILE.read_text())
            self.query_one("#name_input", Input).value = self.loaded_state.get("name_input", "")
            self.query_one("#year_input", Input).value = self.loaded_state.get("year_input", str(END_YEAR - 1))
            self.query_one("#n_input", Input).value = self.loaded_state.get("n_input", "10")
            if sex_button_id := self.loaded_state.get("sex_filter"):
                self.query_one(f"#{sex_button_id}", RadioButton).value = True
            self.query_one("#normalize_data", Checkbox).value = self.loaded_state.get("normalize_data", False)
            if control_tab_id := self.loaded_state.get("control_tab"):
                self.query_one("#control-tabs").active = control_tab_id
            if compare_names := self.loaded_state.get("compare_names"):
                for i in self.query(".comparison-input"): i.remove()
                for i, name in enumerate(compare_names):
                    self.query_one("#comparison-inputs").mount(Input(placeholder=f"Name {i+1}", classes="comparison-input", value=name))
            last_view = self.loaded_state.get("active_view")
            if last_view in ("plot", "unique", "movers", "compare", "origins"):
                self.query_one("#plot-container").display = True
                self.query_one("#top-names-table").display = False
            elif last_view in ("table", "enduring", "details", "similar", "phonetic"):
                self.query_one("#plot-container").display = False
                self.query_one("#top-names-table").display = True
            self._update_compare_buttons()
        except (json.JSONDecodeError, KeyError) as e:
            self.status_widget.update(f"[bold red]Error loading state: {e}[/]")
            self.loaded_state = None

    def _setup_plot_axes(self, plt, data, y_column, y_label, is_normalized):
        """Helper to configure plot axes and ticks consistently."""
        plt.xlabel("Year")
        plt.ylabel(y_label)
        plt.grid(True, True)

        plt.xlim(START_YEAR, END_YEAR)
        tick_interval = 20
        start_tick = math.ceil(START_YEAR / tick_interval) * tick_interval
        year_ticks = [int(yr) for yr in range(start_tick, END_YEAR + 1, tick_interval)]
        plt.xticks(year_ticks)

        max_y = data[y_column].max()
        if max_y > 0:
            if is_normalized:
                y_ticks = [i * max_y / 4 for i in range(5)] if max_y > 0 else []
                y_labels = [f"{tick:.3f}%" for tick in y_ticks]
                plt.yticks(y_ticks, y_labels)
            else:
                if max_y < 100: interval = 10
                elif max_y < 1000: interval = 100
                elif max_y < 10000: interval = 1000
                elif max_y < 50000: interval = 5000
                else: interval = 10000
                count_ticks = [int(i) for i in range(0, int(max_y) + interval, interval)]
                count_labels = [f"{tick:,}" for tick in count_ticks]
                plt.yticks(count_ticks, count_labels)

    def _draw_welcome_plot(self) -> None:
        plot = self.query_one(PlotextPlot)
        plt = plot.plt
        plt.clear_data()
        plt.title("Welcome to Name Frequency Analysis!")
        plt.xlabel("Select an analysis from the left panel. Press '?' for help.")
        plt.grid(True, True)
        plt.build()
        plot.refresh()
        self.query_one("#plot-details-table").display = False

    def _draw_plot(self) -> None:
        self.current_view = "plot"
        name = self.query_one("#name_input", Input).value.strip()
        if not name:
            self.status_widget.update("[bold yellow]Please enter a name to plot.[/]")
            return
        sex_filter = self.query_one(RadioSet).pressed_button.label.plain
        name_data = self.data_analyzer.get_name_history(name, sex_filter)
        if name_data is None or name_data['Count'].sum() == 0:
            self.status_widget.update(f"[bold yellow]Name '{name}' not found for the selected sex.[/]")
            self.query_one("#plot-details-table").display = False
            self._draw_welcome_plot()
            return
        is_normalized = self.query_one("#normalize_data", Checkbox).value
        y_column = "Percentage" if is_normalized else "Count"
        y_label = "% of Total Births" if is_normalized else "Number of Births"
        plot = self.query_one(PlotextPlot)
        plt = plot.plt
        plt.clear_data()

        for sex, group in name_data.groupby("Sex"):
            if group['Count'].sum() > 0:
                plt.plot(group["Year"], group[y_column], label=f"{name.title()} ({sex})")
        
        plt.title(f"Popularity of the Name '{name.title()}'")
        self._setup_plot_axes(plt, name_data, y_column, y_label, is_normalized)
        plt.build()
        plot.refresh()

        details_table = self.query_one("#plot-details-table")
        details_table.display = True
        details_table.clear(columns=True)
        pivoted = name_data[name_data['Count'] > 0].pivot_table(index='Year', columns='Sex', values=['Count', 'Percentage']).fillna(0).sort_index(ascending=False)
        details_table.add_columns("Year", "F Count", "F %", "M Count", "M %")
        for year, row in pivoted.iterrows():
            details_table.add_row(
                str(int(year)),
                f"{int(row.get(('Count', 'F'), 0)):,}", f"{row.get(('Percentage', 'F'), 0):.4f}%",
                f"{int(row.get(('Count', 'M'), 0)):,}", f"{row.get(('Percentage', 'M'), 0):.4f}%"
            )
        self.status_widget.update(f"Showing plot and details for '{name.title()}'.")
        self._save_state()

    def _get_name_details_with_worker(self) -> None:
        self.current_view = "details"
        name = self.query_one("#name_input").value.strip()
        if not name:
            self.status_widget.update("[bold yellow]Please enter a name to get details.[/]")
            return
        self.status_widget.update("Calculating name details...")
        self.query_one("#loading-overlay").display = True
        self.calculate_name_details(name)

    @work(exclusive=True, thread=True)
    def calculate_name_details(self, name: str) -> None:
        sex_filter = self.query_one(RadioSet).pressed_button.label.plain
        details = self.data_analyzer.get_name_details(name, sex_filter)
        self.call_from_thread(self._display_name_details, details, name)

    def _display_name_details(self, details: Optional[dict], name: str) -> None:
        self.query_one("#loading-overlay").display = False
        table = self.query_one("#top-names-table")
        table.clear(columns=True)
        if details is None:
            table.add_column("Error")
            table.add_row(f"No data found for the name '{name}'.")
            self.status_widget.update(f"[bold red]Could not find details for '{name}'.[/]")
            return
        self.status_widget.update(f"Showing details for '{name.title()}'.")
        table.add_column("Statistic", key="stat", width=25)
        table.add_column("Value", key="value")
        table.add_row("Total Births", f"{details['Total Births']:,}")
        table.add_row("First Appearance", str(details['First Appearance']))
        table.add_row("Peak Year", f"{details['Peak Year']} ({details['Peak Year Count']:,} births)")
        table.add_row("Most Popular Decade", details['Most Popular Decade'])
        ranks = details.get("All-Time Rank", {})
        if not ranks: table.add_row("All-Time Rank", "N/A")
        else:
            for sex, rank in ranks.items():
                table.add_row(f"All-Time Rank ({'F' if sex == 'F' else 'M'})", f"#{rank:,}")
        
        # Display Name Signature
        table.add_row("", "") # Separator
        table.add_row(f"[bold]Peak Year Signature[/]", f"({details.get('Peak Year')})")
        top_males = details.get("Peak Year Signature (Male)", [])
        top_females = details.get("Peak Year Signature (Female)", [])
        if top_males or top_females:
            table.add_row("[u]Top Male Names[/]", "[u]Top Female Names[/]")
            for i in range(max(len(top_males), len(top_females))):
                male_name = f"{i+1}. {top_males[i]}" if i < len(top_males) else ""
                female_name = f"{i+1}. {top_females[i]}" if i < len(top_females) else ""
                table.add_row(male_name, female_name)

        self._save_state()
    
    def _find_similar_names_with_worker(self) -> None:
        self.current_view = "similar"
        name = self.query_one("#name_input").value.strip()
        if not name:
            self.status_widget.update("[bold yellow]Please enter a name to find similar names.[/]")
            return
        self.status_widget.update(f"Finding names with similar spelling to '{name}'...")
        self.query_one("#loading-overlay").display = True
        self.calculate_similar_names(name)

    @work(exclusive=True, thread=True)
    def calculate_similar_names(self, name: str) -> None:
        sex_filter = self.query_one(RadioSet).pressed_button.label.plain
        similar_names, original_count, original_first_year = self.data_analyzer.get_similar_names(name, sex_filter)
        self.call_from_thread(self._display_similar_names, similar_names, name, original_count, original_first_year)

    def _display_similar_names(self, results: Optional[pd.DataFrame], name: str, original_count: int, original_first_year: Optional[int]) -> None:
        self.query_one("#loading-overlay").display = False
        table = self.query_one("#top-names-table")
        table.clear(columns=True)
        self.status_widget.update(f"Showing names with similar spelling to '{name.title()}'.")
        
        table.add_column("Similar Name", key="name")
        table.add_column("Sex", key="sex")
        table.add_column("First Year", key="first_year")
        table.add_column("Distance", key="distance")
        table.add_column("Total Births", key="count")
        table.add_column("Popularity vs. Original", key="ratio")
        
        original_first_year_str = str(original_first_year) if original_first_year else "N/A"
        table.add_row(
            f"[bold yellow]{name.title()}[/]", 
            "[yellow](Original)[/]", 
            f"[yellow]{original_first_year_str}[/]",
            "[yellow]0[/]", 
            f"[yellow]{original_count:,}[/]", 
            "[yellow]100%[/]"
        )

        if results is None or results.empty:
            table.add_row("---", "---", "---", "---", "---", "---")
            table.add_row("No similar names found.")
        else:
            for row in results.itertuples():
                ratio_str = f"{row.PopularityRatio:.1%}" if pd.notna(row.PopularityRatio) else "N/A"
                first_year_str = str(int(row.FirstYear)) if pd.notna(row.FirstYear) else "N/A"
                table.add_row(row.Name, row.Sex, first_year_str, str(row.Distance), f"{row.Count:,}", ratio_str)
        self._save_state()

    def _find_phonetic_names_with_worker(self) -> None:
        self.current_view = "phonetic"
        name = self.query_one("#name_input").value.strip()
        if not name:
            self.status_widget.update("[bold yellow]Please enter a name to find phonetic matches.[/]")
            return
        self.status_widget.update(f"Finding names that sound like '{name}'...")
        self.query_one("#loading-overlay").display = True
        self.calculate_phonetic_names(name)

    @work(exclusive=True, thread=True)
    def calculate_phonetic_names(self, name: str) -> None:
        sex_filter = self.query_one(RadioSet).pressed_button.label.plain
        results, original_count, original_first_year = self.data_analyzer.get_phonetically_similar_names(name, sex_filter)
        self.call_from_thread(self._display_phonetic_names, results, name, original_count, original_first_year)
    
    def _display_phonetic_names(self, results: Optional[pd.DataFrame], name: str, original_count: int, original_first_year: Optional[int]) -> None:
        self.query_one("#loading-overlay").display = False
        table = self.query_one("#top-names-table")
        table.clear(columns=True)
        self.status_widget.update(f"Showing names that sound like '{name.title()}'.")
        
        table.add_column("Phonetic Match", key="name")
        table.add_column("Sex", key="sex")
        table.add_column("First Year", key="first_year")
        table.add_column("Total Births", key="count")
        
        original_first_year_str = str(original_first_year) if original_first_year else "N/A"
        table.add_row(
            f"[bold yellow]{name.title()}[/]",
            "[yellow](Original)[/]",
            f"[yellow]{original_first_year_str}[/]",
            f"[yellow]{original_count:,}[/]"
        )

        if results is None or results.empty:
            table.add_row("No phonetic matches found.")
        else:
            for row in results.itertuples():
                first_year_str = str(int(row.FirstYear)) if pd.notna(row.FirstYear) else "N/A"
                table.add_row(row.Name, row.Sex, first_year_str, f"{row.Count:,}")
        self._save_state()
    
    def _draw_name_origins_with_worker(self):
        self.current_view = "origins"
        self.status_widget.update("Calculating name origins by decade...")
        self.query_one("#loading-overlay").display = True
        self.calculate_name_origins()

    @work(exclusive=True, thread=True)
    def calculate_name_origins(self):
        sex_filter = self.query_one(RadioSet).pressed_button.label.plain
        origins_data = self.data_analyzer.get_name_origins_by_decade(sex_filter)
        self.call_from_thread(self._display_name_origins, origins_data)

    def _display_name_origins(self, origins_data: Optional[pd.DataFrame]):
        self.query_one("#loading-overlay").display = False
        plot = self.query_one(PlotextPlot)
        plt = plot.plt
        plt.clear_data()

        if origins_data is None or origins_data.empty:
            self.status_widget.update("[bold red]Could not calculate name origins.[/]")
            self._draw_welcome_plot()
            return
        
        sex_filter = self.query_one(RadioSet).pressed_button.label.plain
        self.status_widget.update(f"Showing new {sex_filter.lower()} name origins by decade.")

        # Create bar chart
        decade_labels = [f"{decade}s" for decade in origins_data['Decade']]
        plt.bar(decade_labels, origins_data['NewNameCount'])
        plt.title("New Unique Names Introduced per Decade")
        plt.xlabel("Decade")
        plt.ylabel("Number of New Names")
        plt.build()
        plot.refresh()

        # Display details in table
        details_table = self.query_one("#plot-details-table")
        details_table.display = True
        details_table.clear(columns=True)
        details_table.add_column("Decade", key="decade")
        details_table.add_column("New Names Introduced", key="new_names")
        for row in origins_data.itertuples():
            details_table.add_row(f"{row.Decade}s", f"{row.NewNameCount:,}")
        self._save_state()


    def _draw_unique_names_plot(self) -> None:
        self.current_view = "unique"
        sex_label = self.query_one(RadioSet).pressed_button.label.plain
        unique_names = self.data_analyzer.get_unique_names_per_year(sex_label)
        if unique_names is None or unique_names.empty:
            self.status_widget.update("[bold red]Could not calculate unique names.[/]")
            return
        self.status_widget.update(f"Plotting unique {sex_label.lower()} names per year...")
        plot = self.query_one(PlotextPlot)
        plt = plot.plt
        plt.clear_data()
        plt.plot(unique_names["Year"], unique_names["Name"])
        plt.title(f"Unique {sex_label} Names Per Year")
        self._setup_plot_axes(plt, unique_names, 'Name', "Number of Unique Names", False)
        plt.build()
        plot.refresh()
        details_table = self.query_one("#plot-details-table")
        details_table.display = True
        details_table.clear(columns=True)
        details_table.add_column("Year", key="Year")
        details_table.add_column(f"Unique {sex_label} Names", key="unique_count")
        for row in unique_names.sort_values(by="Year", ascending=False).itertuples():
            details_table.add_row(str(row.Year), f"{row.Name:,}")
        self._save_state()

    def _get_validated_input(self, input_id: str, default: int) -> int:
        input_widget = self.query_one(f"#{input_id}", Input)
        if input_widget.is_valid:
            return int(input_widget.value)
        else:
            self.status_widget.update(f"[bold yellow]Invalid value in '{input_widget.placeholder}'. Using {default}.[/]")
            input_widget.value = str(default)
            return default

    def _draw_biggest_movers_plot_with_worker(self) -> None:
        self.current_view = "movers"
        self.status_widget.update("Calculating biggest movers...")
        self.query_one("#loading-overlay").display = True
        self.calculate_biggest_movers()

    @work(exclusive=True, thread=True)
    def calculate_biggest_movers(self) -> None:
        n = self._get_validated_input("n_input", 10)
        sex_label = self.query_one(RadioSet).pressed_button.label.plain
        is_normalized = self.query_one("#normalize_data", Checkbox).value
        movers_data = self.data_analyzer.get_biggest_movers(n, sex_label, is_normalized)
        self.call_from_thread(self._display_biggest_movers, movers_data, is_normalized, n)

    def _display_biggest_movers(self, movers_data: dict, is_normalized: bool, n: int) -> None:
        self.query_one("#loading-overlay").display = False
        top_gainers = movers_data.get('gainers')
        top_losers = movers_data.get('losers')
        plot = self.query_one(PlotextPlot)
        plt = plot.plt
        plt.clear_data()
        
        plot_data_list = []
        if top_gainers is not None and not top_gainers.empty:
            gainer_history = self.data_analyzer.get_name_history(top_gainers.iloc[0]['Name'], top_gainers.iloc[0]['Sex'])
            if gainer_history is not None:
                plt.plot(gainer_history["Year"], gainer_history["Count"], label=f"Top Gainer: {top_gainers.iloc[0]['Name']}")
                plot_data_list.append(gainer_history)
        if top_losers is not None and not top_losers.empty:
            loser_history = self.data_analyzer.get_name_history(top_losers.iloc[0]['Name'], top_losers.iloc[0]['Sex'])
            if loser_history is not None:
                plt.plot(loser_history["Year"], loser_history["Count"], label=f"Top Loser: {top_losers.iloc[0]['Name']}")
                plot_data_list.append(loser_history)

        plt.title("Historical Popularity of Biggest Movers")
        if plot_data_list:
            combined_plot_data = pd.concat(plot_data_list)
            self._setup_plot_axes(plt, combined_plot_data, 'Count', "Total Count", False)
        
        plt.build()
        plot.refresh()

        details_table = self.query_one("#plot-details-table")
        details_table.display = True
        details_table.clear(columns=True)
        change_label = 'Change %' if is_normalized else 'Change'
        details_table.add_column("Top Gainers", key="gainer_name")
        details_table.add_column("Sex", key="gainer_sex")
        details_table.add_column("Year", key="gainer_year")
        details_table.add_column(change_label, key="gainer_change")
        details_table.add_column("|", key="separator")
        details_table.add_column("Top Fallers", key="faller_name")
        details_table.add_column("Sex", key="faller_sex")
        details_table.add_column("Year", key="faller_year")
        details_table.add_column(change_label, key="faller_change")
        details_table.columns["separator"].cell_class = "separator-column"
        for i in range(n):
            g_name, g_sex, g_year, g_change = "", "", "", ""
            if top_gainers is not None and i < len(top_gainers):
                gainer = top_gainers.iloc[i]
                g_val = gainer['Change']
                g_str = f"[green]+{g_val:.2f}%[/]" if is_normalized else f"[green]+{g_val:,.0f}[/]"
                g_name, g_sex, g_year, g_change = gainer['Name'], gainer['Sex'], str(gainer['Year']), g_str
            l_name, l_sex, l_year, l_change = "", "", "", ""
            if top_losers is not None and i < len(top_losers):
                loser = top_losers.iloc[i]
                l_val = loser['Change']
                l_str = f"[red]{l_val:.2f}%[/]" if is_normalized else f"[red]{l_val:,.0f}[/]"
                l_name, l_sex, l_year, l_change = loser['Name'], loser['Sex'], str(loser['Year']), l_str
            details_table.add_row(g_name, g_sex, g_year, g_change, "|", l_name, l_sex, l_year, l_change)
        self.status_widget.update("Showing biggest gainers and fallers.")
        self._save_state()

    def _draw_comparison_plot(self) -> None:
        self.current_view = "compare"
        names_to_compare = [inp.value.strip().title() for inp in self.query(".comparison-input") if inp.value.strip()]
        if not names_to_compare:
            self.status_widget.update("[bold yellow]Please enter at least one name to compare.[/]")
            self._draw_welcome_plot()
            self.query_one("#plot-details-table").display = False
            return
        self.status_widget.update(f"Comparing {', '.join(names_to_compare)}...")
        is_normalized = self.query_one("#normalize_data", Checkbox).value
        sex_filter = self.query_one(RadioSet).pressed_button.label.plain
        y_column = "Percentage" if is_normalized else "Count"
        y_label = "% of Total Births" if is_normalized else "Number of Births"
        
        plot = self.query_one(PlotextPlot)
        plt = plot.plt
        plt.clear_data()

        all_name_data_list = [self.data_analyzer.get_name_history(name, sex_filter) for name in names_to_compare]
        all_name_data_list = [df for df in all_name_data_list if df is not None and df['Count'].sum() > 0]
        if not all_name_data_list:
            self.status_widget.update(f"[bold yellow]None of the specified names were found.[/]")
            self._draw_welcome_plot()
            self.query_one("#plot-details-table").display = False
            return
        
        combined_data = pd.concat(all_name_data_list)
        for name, group in combined_data.groupby(['Name', 'Sex']):
            plt.plot(group["Year"], group[y_column], label=f"{name[0]} ({name[1]})")

        plt.title(f"Comparison: {', '.join(names_to_compare)}")
        self._setup_plot_axes(plt, combined_data, y_column, y_label, is_normalized)
        plt.build()
        plot.refresh()

        details_table = self.query_one("#plot-details-table")
        details_table.display = True
        details_table.clear(columns=True)

        pivoted = combined_data[combined_data['Count'] > 0].pivot_table(
            index='Year', 
            columns=['Name', 'Sex'], 
            values='Count'
        ).fillna(0)
        
        pivoted.columns = [f"{name} ({sex})" for name, sex in pivoted.columns]
        pivoted = pivoted.sort_index(ascending=False)

        if not pivoted.empty:
            details_table.add_column("Year", key="Year")
            for col_name in pivoted.columns:
                sanitized_key = col_name.replace(" ", "_").replace("(", "").replace(")", "")
                details_table.add_column(col_name, key=sanitized_key)

            for year, row in pivoted.iterrows():
                row_values = [str(int(year))] + [f"{int(count):,}" for count in row]
                details_table.add_row(*row_values)

        self.status_widget.update(f"Showing comparison for: {', '.join(names_to_compare)}")
        self._save_state()

    def _draw_enduring_popularity_with_worker(self) -> None:
        self.current_view = "enduring"
        self.status_widget.update("Calculating enduring popularity...")
        self.query_one("#loading-overlay").display = True
        self.calculate_enduring_popularity()

    @work(exclusive=True, thread=True)
    def calculate_enduring_popularity(self) -> None:
        n = self._get_validated_input("n_input", 10)
        sex_filter = self.query_one(RadioSet).pressed_button.label.plain
        top_enduring = self.data_analyzer.get_enduring_popularity(n, sex_filter)
        self.call_from_thread(self._display_enduring_popularity, top_enduring)

    def _display_enduring_popularity(self, top_enduring: Optional[pd.DataFrame]) -> None:
        self.query_one("#loading-overlay").display = False
        table = self.query_one("#top-names-table")
        table.clear(columns=True)
        table.add_column("Rank", key="rank")
        table.add_column("Name", key="name")
        table.add_column("Sex", key="sex")
        table.add_column("Avg. Popularity (%)", key="avg_perc")
        if top_enduring is not None and not top_enduring.empty:
            for i, row in enumerate(top_enduring.itertuples(), 1):
                table.add_row(str(i), row.Name, row.Sex, f"{row.Percentage:.5f}%")
        else:
            table.add_row("Could not calculate enduring popularity.")
        self.status_widget.update("Showing names with the most enduring popularity.")
        self._save_state()

    def _populate_table_with_worker(self) -> None:
        self.current_view = "table"
        if not self.query_one("#year_input").is_valid:
            self.status_widget.update(f"[bold red]Invalid year. Must be between {START_YEAR} and {END_YEAR}.[/]")
            return
        self.status_widget.update("Calculating top names...")
        self.query_one("#loading-overlay").display = True
        self.calculate_top_names()

    @work(exclusive=True, thread=True)
    def calculate_top_names(self) -> None:
        year = self._get_validated_input("year_input", END_YEAR - 1)
        n = self._get_validated_input("n_input", 10)
        sex_label = self.query_one(RadioSet).pressed_button.label.plain
        results = self.data_analyzer.get_top_names(year, n, sex_label)
        self.call_from_thread(self._display_top_names, results, year, n, sex_label)

    def _display_top_names(self, results: dict, year: int, n: int, sex_label: str) -> None:
        table = self.query_one("#top-names-table")
        table.clear(columns=True)
        if not results:
             table.add_column("Info")
             table.add_row(f"No data available for the year {year}.")
             self.query_one("#loading-overlay").display = False
             return
        
        def get_move_str(mvmt):
            if pd.isna(mvmt) or mvmt == 'New': return "[blue]New[/]"
            mvmt = int(mvmt)
            if mvmt == 0: return "-"
            return f"[green]+{mvmt}[/]" if mvmt > 0 else f"[red]{mvmt}[/]"

        if sex_label == "Both":
            table.add_column("Rank", key="rank_f"); table.add_column("Mvmt", key="mvmt_f"); table.add_column("Female", key="name_f"); table.add_column("Count", key="count_f"); table.add_column("%", key="perc_f"); table.add_column("|", key="separator"); table.add_column("Rank", key="rank_m"); table.add_column("Mvmt", key="mvmt_m"); table.add_column("Male", key="name_m"); table.add_column("Count", key="count_m"); table.add_column("%", key="perc_m")
            table.columns["separator"].cell_class = "separator-column"
            top_f = results.get('female', pd.DataFrame()); top_m = results.get('male', pd.DataFrame())
            for i in range(max(len(top_f), len(top_m))):
                f_row = top_f.iloc[i] if i < len(top_f) else None; m_row = top_m.iloc[i] if i < len(top_m) else None
                table.add_row(
                    str(i + 1) if f_row is not None else "", get_move_str(f_row['Mvmt']) if f_row is not None else "", f_row['Name'] if f_row is not None else "", f"{f_row['Count']:,}" if f_row is not None else "", f"{f_row['Percentage']:.4f}%" if f_row is not None else "", "|",
                    str(i + 1) if m_row is not None else "", get_move_str(m_row['Mvmt']) if m_row is not None else "", m_row['Name'] if m_row is not None else "", f"{m_row['Count']:,}" if m_row is not None else "", f"{m_row['Percentage']:.4f}%" if m_row is not None else ""
                )
        else:
            table.add_column("Rank", key="rank"); table.add_column("Mvmt", key="mvmt"); table.add_column("Name", key="name"); table.add_column("Sex", key="sex"); table.add_column("Count", key="count"); table.add_column("% of Total", key="perc")
            top_names = results.get(sex_label.lower(), pd.DataFrame())
            for i, row in top_names.iterrows():
                table.add_row(str(i + 1), get_move_str(row.Mvmt), row.Name, row.Sex, f"{row.Count:,}", f"{row.Percentage:.4f}%")
        self.status_widget.update(f"Showing top {n} names for {year} ({sex_label}).")
        self.query_one("#loading-overlay").display = False
        self._save_state()

    def _update_compare_buttons(self) -> None:
        num_inputs = len(self.query(".comparison-input"))
        self.query_one("#remove_name_field_button").disabled = num_inputs <= 2

    def _update_year_nav_buttons(self):
        year_input = self.query_one("#year_input", Input)
        if year_input.is_valid:
            year = int(year_input.value)
            self.query_one("#prev_year_button").disabled = year <= START_YEAR
            self.query_one("#next_year_button").disabled = year >= END_YEAR
        else:
            self.query_one("#prev_year_button").disabled = True
            self.query_one("#next_year_button").disabled = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self.data_analyzer.df is None and event.button.id not in ("quit", "toggle_dark"):
            self.status_widget.update("[bold yellow]Please wait for the data to load.[/]")
            return
        
        actions = {
            "plot_button": (True, self._draw_plot), "get_details_button": (False, self._get_name_details_with_worker),
            "find_similar_button": (False, self._find_similar_names_with_worker),
            "find_phonetic_button": (False, self._find_phonetic_names_with_worker),
            "top_names_button": (False, self._populate_table_with_worker), "unique_names_button": (True, self._draw_unique_names_plot),
            "biggest_movers_button": (True, self._draw_biggest_movers_plot_with_worker), "enduring_popularity_button": (False, self._draw_enduring_popularity_with_worker),
            "plot_comparison_button": (True, self._draw_comparison_plot),
            "name_origins_button": (True, self._draw_name_origins_with_worker)
        }
        
        if event.button.id in actions:
            show_plot, action = actions[event.button.id]
            self.query_one("#plot-container").display = show_plot
            self.query_one("#top-names-table").display = not show_plot
            self.call_later(action)
        elif event.button.id == "add_name_field_button":
            new_input = Input(placeholder=f"Name {len(self.query('.comparison-input')) + 1}", classes="comparison-input")
            self.query_one("#comparison-inputs").mount(new_input)
            new_input.focus()
            self._update_compare_buttons()
        elif event.button.id == "remove_name_field_button":
            if len(self.query(".comparison-input")) > 2:
                self.query(".comparison-input").last().remove()
            self._update_compare_buttons()
        elif event.button.id in ("prev_year_button", "next_year_button"):
            year_input = self.query_one("#year_input", Input)
            if year_input.is_valid:
                current_year = int(year_input.value)
                offset = -1 if event.button.id == "prev_year_button" else 1
                year_input.value = str(current_year + offset)
                self.query_one("#top_names_button").press()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "year_input":
            self._update_year_nav_buttons()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self.data_analyzer.df is None: return
        action_map = {
            "name_input": "#get_details_button",
            "year_input": "#top_names_button", "n_input": "#top_names_button",
        }
        if event.input.id in action_map:
            self.query_one(action_map[event.input.id]).press()
        elif "comparison-input" in event.input.classes:
            self.query_one("#plot_comparison_button").press()

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        if self.data_analyzer.df is None: return
        
        col_key = event.cell_key.column_key.value

        name_keys = {"name_f", "name_m", "name", "gainer_name", "faller_name"}
        year_keys = {"Year", "gainer_year", "faller_year"}

        if col_key in name_keys:
            if selected_name := str(event.value):
                self.query_one("#name_input").value = selected_name
                self.query_one("#control-tabs").active = "single-name-controls"
                self.query_one("#get_details_button").press()
        elif col_key in year_keys:
            self.query_one("#year_input").value = str(event.value)
            self.query_one("#control-tabs").active = "top-names-controls"
            self.query_one("#top_names_button").press()

if __name__ == "__main__":
    app = NameAnalysisApp()
    app.run()
