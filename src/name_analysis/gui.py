#!/usr/bin/env python

import flet as ft
import matplotlib
matplotlib.use("Agg")  # Use the Agg backend for thread-safe plotting
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .data_analyzer import DataAnalyzer, START_YEAR, END_YEAR
import re
from io import BytesIO
import base64
import json
from pathlib import Path

# --- Configuration ---
STATE_FILE = Path("app_state.json")

class NameAnalysisGUI:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "Name Frequency Analysis (GUI)"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.window_width = 1400
        self.page.window_height = 900
        
        # Set the base theme colors
        self.page.theme = ft.Theme(color_scheme_seed=ft.Colors.BLUE_GREY)
        self.page.dark_theme = ft.Theme(color_scheme_seed=ft.Colors.BLUE_GREY)

        self.analyzer = DataAnalyzer()
        self.current_view = "welcome"  # Tracks the current analysis being displayed
        self.loaded_state = None  # Stores state loaded from file

        # --- UI Control References ---
        self.plot_image = ft.Image(expand=True)
        self.data_table = ft.DataTable(columns=[], rows=[], expand=True)
        self.results_stack = ft.Stack(expand=True)
        self.progress_ring = ft.ProgressRing(width=32, height=32)
        self.status_text = ft.Text("Loading...")
        
        # --- Load State & Set Handlers BEFORE Building UI ---
        self._load_state()  # Load state from file before building controls
        self.page.on_window_event = self.on_window_event

        # --- Build UI ---
        self.build_ui()

        # --- Initial Data Load ---
        self.page.run_thread(self.load_data_worker)

    def on_window_event(self, e):
        """Saves the application state when the window is closed."""
        if e.data == "close":
            self._save_state()
            self.page.window_destroy()

    def _save_state(self):
        """Gathers UI control values and saves them to a TUI-compatible JSON file."""
        try:
            gui_to_tui_sex_map = {"Male": "sex_male", "Female": "sex_female", "Both": "sex_both"}
            index_to_tab_id_map = {0: "single-name-controls", 1: "top-names-controls", 2: "analysis-controls", 3: "compare-controls"}
            sex_filter_to_save = gui_to_tui_sex_map.get(self.sex_filter.value, "sex_both")
            tab_id_to_save = index_to_tab_id_map.get(self.control_tabs.selected_index, "single-name-controls")

            state = {
                "name_input": self.name_input.value,
                "year_input": self.year_input.value,
                "n_input": self.n_input.value,
                "sex_filter": sex_filter_to_save,
                "normalize_data": self.normalize_checkbox.value,
                "show_axis_lines": self.axis_lines_checkbox.value,
                "control_tab": tab_id_to_save,
                "active_view": self.current_view,
                "compare_names": [inp.value for inp in self.compare_name_inputs.controls]
            }
            STATE_FILE.write_text(json.dumps(state, indent=4))
        except Exception as e:
            print(f"Error saving state: {e}")

    def _load_state(self):
        """Loads state from JSON file."""
        if not STATE_FILE.exists(): return
        try:
            self.loaded_state = json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading state file: {e}")
            self.loaded_state = None

    def build_ui(self):
        """Builds the main UI structure, including loading and main views."""
        self.loading_container = ft.Column([self.progress_ring, self.status_text], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True, visible=True)
        self.main_container = self.build_full_app_layout()
        self.main_container.visible = False
        self.page.add(self.loading_container, self.main_container)
        self.page.update()

    def load_data_worker(self):
        """Worker thread to load data and update UI when done."""
        for status in self.analyzer.download_and_extract_data(): self.status_text.value = status; self.page.update()
        status = self.analyzer.load_data()
        if self.analyzer.df is not None:
            self.status_text.value = "Data loaded successfully. Ready to analyze."
            self.loading_container.visible = False; self.main_container.visible = True
            if self.loaded_state: self.page.run_thread(self._restore_last_view)
        else:
            self.progress_ring.visible = False; self.status_text.value = f"Failed to load data. Error: {status}"
        self.page.update()

    def _restore_last_view(self):
        """Re-runs the last executed analysis based on the loaded state."""
        import time; time.sleep(0.5)
        view_to_restore = self.loaded_state.get("active_view")
        if not view_to_restore or view_to_restore == "welcome": return
        view_map = {
            "details": self.get_details_worker, "plot": self.plot_history_worker, "plot_history": self.plot_history_worker,
            "table": self.get_top_names_worker, "top_names": self.get_top_names_worker,
            "similar": self.find_similar_worker, "similar_spelling": self.find_similar_worker,
            "phonetic": self.find_phonetic_worker, "similar_sound": self.find_phonetic_worker,
            "unique": self.unique_names_worker, "unique_names": self.unique_names_worker,
            "movers": self.biggest_movers_worker, "biggest_movers": self.biggest_movers_worker,
            "enduring": self.enduring_popularity_worker, "enduring_popularity": self.enduring_popularity_worker,
            "origins": self.name_origins_worker, "name_origins": self.name_origins_worker,
            "compare": self.comparison_plot_worker, "comparison_plot": self.comparison_plot_worker,
        }
        worker_func = view_map.get(view_to_restore)
        if worker_func: worker_func()
            
    def _plot_option_changed(self, e=None):
        """Refreshes the current plot if a plot-related option changes."""
        plot_views = ["plot_history", "unique_names", "biggest_movers", "name_origins", "comparison_plot"]
        if self.current_view in plot_views:
            view_map = {
                "plot_history": self.run_plot_history_worker,
                "unique_names": self.run_unique_names_worker,
                "biggest_movers": self.run_biggest_movers_worker,
                "name_origins": self.run_name_origins_worker,
                "comparison_plot": self.run_comparison_plot_worker
            }
            worker_to_run = view_map.get(self.current_view)
            if worker_to_run: worker_to_run()

    def build_full_app_layout(self) -> ft.Row:
        """Builds and returns the main application layout control."""
        state = self.loaded_state or {}
        tui_to_gui_sex_map = {"sex_male": "Male", "sex_female": "Female", "sex_both": "Both"}
        sex_from_state = state.get("sex_filter", "sex_both"); sex_value_to_load = tui_to_gui_sex_map.get(sex_from_state, "Both")
        tab_id_to_index_map = {"single-name-controls": 0, "top-names-controls": 1, "analysis-controls": 2, "compare-controls": 3}
        tab_id_from_state = state.get("control_tab", "single-name-controls"); tab_index_to_load = tab_id_to_index_map.get(tab_id_from_state, 0)
        
        self.name_input = ft.TextField(label="Name", on_submit=self.run_get_details_worker, value=state.get("name_input", ""), border_color=ft.Colors.OUTLINE, focused_border_color=ft.Colors.PRIMARY, focused_border_width=2)
        self.year_input = ft.TextField(label="Year", value=state.get("year_input", str(END_YEAR - 1)), width=100, on_submit=self.run_get_top_names_worker, border_color=ft.Colors.OUTLINE, focused_border_color=ft.Colors.PRIMARY, focused_border_width=2)
        self.n_input = ft.TextField(label="Num Results", value=state.get("n_input", "10"), width=100, border_color=ft.Colors.OUTLINE, focused_border_color=ft.Colors.PRIMARY, focused_border_width=2)
        self.sex_filter = ft.RadioGroup(content=ft.Row([ft.Radio(value="Both", label="Both"), ft.Radio(value="Male", label="Male"), ft.Radio(value="Female", label="Female")]), value=sex_value_to_load)
        self.normalize_checkbox = ft.Checkbox(label="Normalize Plot Data", value=state.get("normalize_data", False), on_change=self._plot_option_changed)
        self.axis_lines_checkbox = ft.Checkbox(label="Show Plot Grid", value=state.get("show_axis_lines", True), on_change=self._plot_option_changed)
        
        compare_names_list = state.get("compare_names", ["", ""])
        self.compare_name_inputs = ft.Column([ft.TextField(label=f"Name {i+1}", value=name, border_color=ft.Colors.OUTLINE, focused_border_color=ft.Colors.PRIMARY, focused_border_width=2) for i, name in enumerate(compare_names_list)])

        self.prev_year_button = ft.IconButton(icon=ft.Icons.KEYBOARD_ARROW_LEFT, on_click=self.change_year_clicked, data=-1)
        self.next_year_button = ft.IconButton(icon=ft.Icons.KEYBOARD_ARROW_RIGHT, on_click=self.change_year_clicked, data=1)
        
        self.control_tabs = ft.Tabs(selected_index=tab_index_to_load, animation_duration=300,
            tabs=[
                ft.Tab(text="Single Name", icon=ft.Icons.PERSON, content=ft.Container(content=ft.Column([ self.name_input, ft.Row([ ft.ElevatedButton("Get Details", icon=ft.Icons.INFO, on_click=self.run_get_details_worker, expand=True), ft.ElevatedButton("Plot History", icon=ft.Icons.SHOW_CHART, on_click=self.run_plot_history_worker, expand=True), ]), ft.Row([ ft.ElevatedButton("Similar by Spelling", icon=ft.Icons.SPELLCHECK, on_click=self.run_find_similar_worker, expand=True), ft.ElevatedButton("Similar by Sound", icon=ft.Icons.RECORD_VOICE_OVER, on_click=self.run_find_phonetic_worker, expand=True), ])]), padding=ft.padding.only(top=10))),
                ft.Tab(text="Top Names", icon=ft.Icons.FORMAT_LIST_NUMBERED, content=ft.Container(content=ft.Column([ ft.Row([self.year_input, self.n_input], alignment=ft.MainAxisAlignment.SPACE_BETWEEN), ft.Row([self.prev_year_button, self.next_year_button], alignment=ft.MainAxisAlignment.CENTER), ft.ElevatedButton("Get Top Names", icon=ft.Icons.SEARCH, on_click=self.run_get_top_names_worker), ]), padding=ft.padding.only(top=10))),
                ft.Tab(text="Analysis", icon=ft.Icons.INSIGHTS, content=ft.Container(content=ft.Column([ ft.ElevatedButton("Unique Names per Year", icon=ft.Icons.NEW_RELEASES, on_click=self.run_unique_names_worker), ft.ElevatedButton("Biggest Movers", icon=ft.Icons.TRENDING_UP, on_click=self.run_biggest_movers_worker), ft.ElevatedButton("Enduring Popularity", icon=ft.Icons.FAVORITE, on_click=self.run_enduring_popularity_worker), ft.ElevatedButton("Name Origins by Decade", icon=ft.Icons.HISTORY_EDU, on_click=self.run_name_origins_worker), ]), padding=ft.padding.only(top=10))),
                ft.Tab(text="Compare", icon=ft.Icons.COMPARE_ARROWS, content=ft.Container(content=ft.Column([ ft.Text("Enter names to compare:"), self.compare_name_inputs, ft.Row([ ft.ElevatedButton("Add Name", icon=ft.Icons.ADD, on_click=self.add_compare_name), ft.ElevatedButton("Plot Comparison", icon=ft.Icons.COMPARE, on_click=self.run_comparison_plot_worker) ])]), padding=ft.padding.only(top=10))),
            ], expand=1)

        controls_container = ft.Container(content=ft.Column([ ft.Row([ft.Text("Controls", size=24, weight=ft.FontWeight.BOLD)]), ft.Container(content=self.control_tabs, expand=True), ft.Column([ ft.Divider(), ft.Text("Options", size=20, weight=ft.FontWeight.BOLD), self.sex_filter, self.normalize_checkbox, self.axis_lines_checkbox ])]), width=350, padding=ft.padding.all(10), border=ft.border.only(right=ft.BorderSide(1, ft.Colors.OUTLINE)))
        self.results_stack.controls = [self.plot_image, self.data_table]; self.plot_image.visible = False; self.data_table.visible = True
        self.data_table.columns = [ft.DataColumn(ft.Text("Info"))]; self.data_table.rows = [ft.DataRow(cells=[ft.DataCell(ft.Text("Select an analysis and click a button to see results."))])]
        main_area = ft.Container(content=ft.Column([ ft.Row([ft.Text("Results", size=24, weight=ft.FontWeight.BOLD)]), self.results_stack, ft.Row([ft.Icon(ft.Icons.INFO_OUTLINE), self.status_text]) ], expand=True, horizontal_alignment=ft.CrossAxisAlignment.CENTER), expand=True, padding=ft.padding.all(10))
        return ft.Row([controls_container, main_area], expand=True, vertical_alignment=ft.CrossAxisAlignment.STRETCH)
    
    def add_compare_name(self, e):
        new_field = ft.TextField(label=f"Name {len(self.compare_name_inputs.controls) + 1}", border_color=ft.Colors.OUTLINE, focused_border_color=ft.Colors.PRIMARY, focused_border_width=2)
        self.compare_name_inputs.controls.append(new_field); self.page.update()

    def run_get_details_worker(self, e=None): self.page.run_thread(self.get_details_worker)
    def run_plot_history_worker(self, e=None): self.page.run_thread(self.plot_history_worker)
    def run_get_top_names_worker(self, e=None): self.page.run_thread(self.get_top_names_worker)
    def run_find_similar_worker(self, e=None): self.page.run_thread(self.find_similar_worker)
    def run_find_phonetic_worker(self, e=None): self.page.run_thread(self.find_phonetic_worker)
    def run_unique_names_worker(self, e=None): self.page.run_thread(self.unique_names_worker)
    def run_biggest_movers_worker(self, e=None): self.page.run_thread(self.biggest_movers_worker)
    def run_enduring_popularity_worker(self, e=None): self.page.run_thread(self.enduring_popularity_worker)
    def run_name_origins_worker(self, e=None): self.page.run_thread(self.name_origins_worker)
    def run_comparison_plot_worker(self, e=None): self.page.run_thread(self.comparison_plot_worker)
        
    def change_year_clicked(self, e):
        try:
            current_year = int(self.year_input.value); new_year = current_year + e.control.data
            if START_YEAR <= new_year <= END_YEAR: self.year_input.value = str(new_year); self.run_get_top_names_worker(e)
        except ValueError: pass

    def get_details_worker(self):
        self.current_view = "details"; name = self.name_input.value.strip()
        if not name: self.status_text.value = "Please enter a name."; self.page.update(); return
        details = self.analyzer.get_name_details(name, self.sex_filter.value); self.display_name_details(details, name)
        
    def plot_history_worker(self):
        self.current_view = "plot_history"; name = self.name_input.value.strip()
        if not name: self.status_text.value = "Please enter a name."; self.page.update(); return
        name_data = self.analyzer.get_name_history(name, self.sex_filter.value); self.display_plot_history(name_data, name)
        
    def get_top_names_worker(self):
        self.current_view = "top_names"
        try: year, n = int(self.year_input.value), int(self.n_input.value)
        except ValueError: self.status_text.value = "Year and Num Results must be integers."; self.page.update(); return
        top_names = self.analyzer.get_top_names(year, n, self.sex_filter.value); self.display_top_names(top_names, year, n)
        
    def find_similar_worker(self):
        self.current_view = "similar_spelling"; name = self.name_input.value.strip()
        if not name: self.status_text.value = "Please enter a name."; self.page.update(); return
        results, count, first_year = self.analyzer.get_similar_names(name, self.sex_filter.value); self.display_similar_names(results, name, count, first_year)
        
    def find_phonetic_worker(self):
        self.current_view = "similar_sound"; name = self.name_input.value.strip()
        if not name: self.status_text.value = "Please enter a name."; self.page.update(); return
        results, count, first_year = self.analyzer.get_phonetically_similar_names(name, self.sex_filter.value); self.display_phonetic_names(results, name, count, first_year)

    def unique_names_worker(self):
        self.current_view = "unique_names"; data = self.analyzer.get_unique_names_per_year(self.sex_filter.value); self.display_unique_names_plot(data)

    def biggest_movers_worker(self):
        self.current_view = "biggest_movers"; n = int(self.n_input.value)
        data = self.analyzer.get_biggest_movers(n, self.sex_filter.value, self.normalize_checkbox.value); self.display_biggest_movers(data, n)

    def enduring_popularity_worker(self):
        self.current_view = "enduring_popularity"; n = int(self.n_input.value)
        data = self.analyzer.get_enduring_popularity(n, self.sex_filter.value); self.display_enduring_popularity(data)

    def name_origins_worker(self):
        self.current_view = "name_origins"; data = self.analyzer.get_name_origins_by_decade(self.sex_filter.value); self.display_name_origins(data)
        
    def comparison_plot_worker(self):
        self.current_view = "comparison_plot"
        names_to_compare = [inp.value.strip().title() for inp in self.compare_name_inputs.controls if inp.value.strip()]
        if not names_to_compare: self.status_text.value = "Please enter at least one name."; self.page.update(); return
        all_name_data = [self.analyzer.get_name_history(name, self.sex_filter.value) for name in names_to_compare]; self.display_comparison_plot(all_name_data, names_to_compare)

    def display_name_details(self, details, name):
        self.show_table()
        if details is None: self.data_table.columns = [ft.DataColumn(ft.Text("Error"))]; self.data_table.rows = [ft.DataRow(cells=[ft.DataCell(ft.Text(f"No data for '{name}'"))])]; self.page.update(); self._save_state(); return
        self.data_table.columns = [ft.DataColumn(ft.Text("Statistic", weight=ft.FontWeight.BOLD)), ft.DataColumn(ft.Text("Value", weight=ft.FontWeight.BOLD))]
        rows, stats = [], {"Total Births": f"{details['Total Births']:,}", "First Appearance": str(details['First Appearance']), "Peak Year": f"{details['Peak Year']} ({details['Peak Year Count']:,} births)", "Most Popular Decade": details['Most Popular Decade']}
        for stat, value in stats.items(): rows.append(ft.DataRow(cells=[ft.DataCell(ft.Text(stat)), ft.DataCell(ft.Text(value))]))
        for sex, rank in details.get("All-Time Rank", {}).items(): rows.append(ft.DataRow(cells=[ft.DataCell(ft.Text(f"All-Time Rank ({sex})")), ft.DataCell(ft.Text(f"#{rank:,}"))]))
        rows.append(ft.DataRow(cells=[ft.DataCell(ft.Container(height=10)), ft.DataCell(ft.Text(""))])); rows.append(ft.DataRow(cells=[ft.DataCell(ft.Text(f"Peak Year Signature ({details['Peak Year']})", weight=ft.FontWeight.BOLD)), ft.DataCell(ft.Text(""))]))
        males, females = details.get('Peak Year Signature (Male)', []), details.get('Peak Year Signature (Female)', [])
        for i in range(max(len(males), len(females))):
            male_name, female_name = (f"{i+1}. {males[i]}" if i < len(males) else ""), (f"{i+1}. {females[i]}" if i < len(females) else "")
            rows.append(ft.DataRow(cells=[ft.DataCell(ft.Text(male_name)), ft.DataCell(ft.Text(female_name))]))
        self.data_table.rows, self.status_text.value = rows, f"Details for {name.title()}"; self.page.update(); self._save_state()

    def display_plot_history(self, name_data, name):
        if name_data is None or name_data['Count'].sum() == 0: self.show_table(); self.data_table.columns = [ft.DataColumn(ft.Text("Error"))]; self.data_table.rows = [ft.DataRow(cells=[ft.DataCell(ft.Text(f"No data for '{name}'"))])]; self.status_text.value = f"No data for '{name}'"; self.page.update(); self._save_state(); return
        y_column, y_label = ("Percentage", "% of Total Births") if self.normalize_checkbox.value else ("Count", "Number of Births")
        fig, ax = self.create_plot()
        for sex, group in name_data.groupby('Sex'):
            if group['Count'].sum() > 0: sns.lineplot(x='Year', y=y_column, data=group, label=f"{name.title()} ({sex})", ax=ax)
        ax.set_title(f"Popularity of '{name.title()}'", fontsize=16); ax.set_xlabel("Year", fontsize=12); ax.set_ylabel(y_label, fontsize=12); ax.legend()
        self.plot_image.src_base64 = self.fig_to_base64(fig); self.show_plot(); self.status_text.value = f"Plot for {name.title()}"; self.page.update(); self._save_state()

    def display_top_names(self, top_names, year, n):
        self.show_table()
        if not top_names: self.data_table.columns = [ft.DataColumn(ft.Text("Info"))]; self.data_table.rows = [ft.DataRow(cells=[ft.DataCell(ft.Text(f"No data for {year}."))])]; self.page.update(); self._save_state(); return
        if self.sex_filter.value == "Both":
            self.data_table.columns = [ft.DataColumn(ft.Text("Rank")), ft.DataColumn(ft.Text("Female")), ft.DataColumn(ft.Text("Count")), ft.DataColumn(ft.Text("|")), ft.DataColumn(ft.Text("Rank")), ft.DataColumn(ft.Text("Male")), ft.DataColumn(ft.Text("Count"))]
            rows, females, males = [], top_names.get('female', pd.DataFrame()), top_names.get('male', pd.DataFrame())
            for i in range(n):
                f_name, f_count = (females.iloc[i]['Name'], f"{females.iloc[i]['Count']:,}") if i < len(females) else ("", "")
                m_name, m_count = (males.iloc[i]['Name'], f"{males.iloc[i]['Count']:,}") if i < len(males) else ("", "")
                rows.append(ft.DataRow(cells=[ft.DataCell(ft.Text(str(i+1))), ft.DataCell(ft.Text(f_name)), ft.DataCell(ft.Text(f_count)), ft.DataCell(ft.Text("|")), ft.DataCell(ft.Text(str(i+1))), ft.DataCell(ft.Text(m_name)), ft.DataCell(ft.Text(m_count))]))
            self.data_table.rows = rows
        else:
            self.data_table.columns = [ft.DataColumn(ft.Text("Rank")), ft.DataColumn(ft.Text("Name")), ft.DataColumn(ft.Text("Count"))]
            rows, names = [], top_names.get(self.sex_filter.value.lower(), pd.DataFrame())
            for i, row in names.iterrows(): rows.append(ft.DataRow(cells=[ft.DataCell(ft.Text(str(i+1))), ft.DataCell(ft.Text(row['Name'])), ft.DataCell(ft.Text(f"{row['Count']:,}"))]))
            self.data_table.rows = rows
        self.status_text.value = f"Top {n} names for {year}"; self.page.update(); self._save_state()

    def display_similar_names(self, results, name, original_count, original_first_year):
        self.show_table(); self.status_text.value = f"Names with similar spelling to '{name.title()}'"
        self.data_table.columns = [ft.DataColumn(ft.Text("Name")), ft.DataColumn(ft.Text("Sex")), ft.DataColumn(ft.Text("First Year")), ft.DataColumn(ft.Text("Distance")), ft.DataColumn(ft.Text("Total Births")), ft.DataColumn(ft.Text("Popularity"))]
        rows = [ft.DataRow(cells=[ft.DataCell(ft.Text(name.title(), weight=ft.FontWeight.BOLD)), ft.DataCell(ft.Text("(Original)", italic=True)), ft.DataCell(ft.Text(str(original_first_year or ''))), ft.DataCell(ft.Text("0")), ft.DataCell(ft.Text(f"{original_count:,}")), ft.DataCell(ft.Text("100%"))])]
        if results is not None and not results.empty:
            for row in results.itertuples():
                ratio = f"{row.PopularityRatio:.1%}" if pd.notna(row.PopularityRatio) else "N/A"
                rows.append(ft.DataRow(cells=[ft.DataCell(ft.Text(row.Name)), ft.DataCell(ft.Text(row.Sex)), ft.DataCell(ft.Text(str(int(row.FirstYear)))), ft.DataCell(ft.Text(str(row.Distance))), ft.DataCell(ft.Text(f"{row.Count:,}")), ft.DataCell(ft.Text(ratio))]))
        self.data_table.rows = rows; self.page.update(); self._save_state()

    def display_phonetic_names(self, results, name, original_count, original_first_year):
        self.show_table(); self.status_text.value = f"Names that sound like '{name.title()}'"
        self.data_table.columns = [ft.DataColumn(ft.Text("Name")), ft.DataColumn(ft.Text("Sex")), ft.DataColumn(ft.Text("First Year")), ft.DataColumn(ft.Text("Total Births"))]
        rows = [ft.DataRow(cells=[ft.DataCell(ft.Text(name.title(), weight=ft.FontWeight.BOLD)), ft.DataCell(ft.Text("(Original)", italic=True)), ft.DataCell(ft.Text(str(original_first_year or ''))), ft.DataCell(ft.Text(f"{original_count:,}"))])]
        if results is not None and not results.empty:
            for row in results.itertuples(): rows.append(ft.DataRow(cells=[ft.DataCell(ft.Text(row.Name)), ft.DataCell(ft.Text(row.Sex)), ft.DataCell(ft.Text(str(int(row.FirstYear)))), ft.DataCell(ft.Text(f"{row.Count:,}"))]))
        self.data_table.rows = rows; self.page.update(); self._save_state()

    def display_unique_names_plot(self, data):
        if data is None or data.empty: self.status_text.value = "Could not calculate unique names."; self.page.update(); self._save_state(); return
        fig, ax = self.create_plot(); sns.lineplot(x='Year', y='Name', data=data, ax=ax)
        ax.set_title("Unique Names Per Year", fontsize=16); ax.set_xlabel("Year"); ax.set_ylabel("Number of Unique Names")
        self.plot_image.src_base64 = self.fig_to_base64(fig); self.status_text.value = "Unique names per year"; self.show_plot(); self.page.update(); self._save_state()

    def display_biggest_movers(self, movers_data, n):
        self.show_table()
        if not movers_data: self.status_text.value = "Could not calculate biggest movers."; self.page.update(); self._save_state(); return
        top_gainers, top_losers, rows = movers_data.get('gainers'), movers_data.get('losers'), []
        self.data_table.columns = [ft.DataColumn(ft.Text("Gainers")), ft.DataColumn(ft.Text("Losers"))]
        for i in range(n):
            gainer_text = f"{top_gainers.iloc[i]['Name']} (+{top_gainers.iloc[i]['Change']:,.0f})" if i < len(top_gainers) else ""
            loser_text = f"{top_losers.iloc[i]['Name']} ({top_losers.iloc[i]['Change']:,.0f})" if i < len(top_losers) else ""
            rows.append(ft.DataRow(cells=[ft.DataCell(ft.Text(gainer_text)), ft.DataCell(ft.Text(loser_text))]))
        self.data_table.rows = rows; self.status_text.value = "Biggest year-over-year movers"; self.page.update(); self._save_state()

    def display_enduring_popularity(self, data):
        self.show_table()
        if data is None or data.empty: self.status_text.value = "Could not calculate enduring popularity."; self.page.update(); self._save_state(); return
        self.data_table.columns = [ft.DataColumn(ft.Text("Rank")), ft.DataColumn(ft.Text("Name")), ft.DataColumn(ft.Text("Sex")), ft.DataColumn(ft.Text("Avg. Pop. (%)"))]
        rows = []
        for i, row in data.iterrows(): rows.append(ft.DataRow(cells=[ft.DataCell(ft.Text(str(i+1))), ft.DataCell(ft.Text(row['Name'])), ft.DataCell(ft.Text(row['Sex'])), ft.DataCell(ft.Text(f"{row['Percentage']:.5f}%"))]))
        self.data_table.rows = rows; self.status_text.value = "Most Enduring Names"; self.page.update(); self._save_state()
    
    def display_name_origins(self, data):
        if data is None or data.empty: self.status_text.value = "Could not calculate name origins."; self.page.update(); self._save_state(); return
        fig, ax = self.create_plot()
        decade_labels = [f"{decade}s" for decade in data['Decade']]
        
        # Address warnings by assigning 'hue' and using modern tick rotation.
        sns.barplot(x=decade_labels, y=data['NewNameCount'], ax=ax, palette="viridis", hue=decade_labels, legend=False)
        
        ax.set_title("New Names Introduced per Decade", fontsize=16)
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel("Decade")
        ax.set_ylabel("Number of New Names")
        
        plt.tight_layout()
        self.plot_image.src_base64 = self.fig_to_base64(fig)
        self.status_text.value = "Name Origins by Decade"
        self.show_plot()
        self.page.update()
        self._save_state()

    def display_comparison_plot(self, all_name_data, names_to_compare):
        if not all_name_data: self.status_text.value = "None of the specified names were found."; self.page.update(); self._save_state(); return
        y_column, y_label = ("Percentage", "% of Total Births") if self.normalize_checkbox.value else ("Count", "Number of Births")
        fig, ax = self.create_plot(); valid_data = [df for df in all_name_data if df is not None]
        if not valid_data: self.status_text.value = "No valid data to plot for the selected names."; self.page.update(); self._save_state(); return
        combined_data = pd.concat(valid_data)
        for name, group in combined_data.groupby(['Name', 'Sex']): sns.lineplot(x='Year', y=y_column, data=group, label=f"{name[0]} ({name[1]})", ax=ax)
        ax.set_title(f"Comparison: {', '.join(names_to_compare)}", fontsize=16); ax.legend()
        self.plot_image.src_base64 = self.fig_to_base64(fig); self.status_text.value = f"Comparison plot for specified names."; self.show_plot(); self.page.update(); self._save_state()

    def show_plot(self): self.plot_image.visible = True; self.data_table.visible = False; self.results_stack.update()
    def show_table(self): self.plot_image.visible = False; self.data_table.visible = True; self.results_stack.update()

    def create_plot(self):
        is_dark = self.page.theme_mode == ft.ThemeMode.DARK
        text_color = "white" if is_dark else "black"

        show_grid = self.axis_lines_checkbox.value
        theme_style = "darkgrid" if is_dark else "whitegrid"
        if not show_grid:
            theme_style = "dark" if is_dark else "white"
        
        sns.set_theme(style=theme_style)

        matplotlib.rcParams.update({
            'text.color': text_color,
            'axes.labelcolor': text_color,
            'xtick.color': text_color,
            'ytick.color': text_color,
            'axes.edgecolor': text_color,
            'axes.titlecolor': text_color,
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        return fig, ax

    def fig_to_base64(self, fig):
        buf = BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", transparent=True); plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

def main(page: ft.Page):
    NameAnalysisGUI(page)

if __name__ == "__main__":
    ft.app(target=main)
