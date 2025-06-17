#!/usr/bin/env python

import flet as ft
from flet.matplotlib_chart import MatplotlibChart
import matplotlib.pyplot as plt
import seaborn as sns
from .data_analyzer import DataAnalyzer

class NameAnalysisGUI:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "Name Frequency Analysis (GUI)"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.window_width = 1200
        self.page.window_height = 800
        self.page.vertical_alignment = ft.MainAxisAlignment.CENTER
        self.page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

        self.analyzer = DataAnalyzer()

        # Build the initial UI
        self.build_ui()

        # Start data loading in the background
        self.page.run_thread(self.load_data_worker)

    def build_ui(self):
        """Builds the main user interface."""
        self.progress_ring = ft.ProgressRing(width=32, height=32)
        self.status_text = ft.Text("Loading data, please wait...")

        self.page.add(
            ft.Column(
                [self.progress_ring, self.status_text],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            )
        )
        self.page.update()

    def load_data_worker(self):
        """Worker thread to load data and update UI when done."""
        for status in self.analyzer.download_and_extract_data():
            self.status_text.value = status
            self.page.update()
        
        status = self.analyzer.load_data()
        self.status_text.value = status
        self.page.update()

        if self.analyzer.df is not None:
            # Data is loaded, build the full interactive UI
            self.page.clean() # Remove the loading indicator
            self.build_full_app()
            self.status_text.value = "Data loaded successfully. Ready to analyze."
            self.page.update()
        else:
            self.progress_ring.visible = False
            self.status_text.value = "Failed to load data. Please check files and restart."
            self.page.update()
            
    def build_full_app(self):
        """Builds the main application layout after data is loaded."""
        # --- Placeholder for the full UI ---
        title = ft.Text("Name Analysis", size=32, weight=ft.FontWeight.BOLD)
        
        # We will replace this with the actual tabbed interface next.
        main_content = ft.Column([
            title,
            ft.Text("Application UI will be built here."),
            self.status_text # Reuse the status text widget
        ], expand=True, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        
        self.page.add(main_content)
        self.page.vertical_alignment = ft.MainAxisAlignment.START
        self.page.horizontal_alignment = ft.CrossAxisAlignment.START

def main(page: ft.Page):
    """Main function to initialize and run the Flet app."""
    NameAnalysisGUI(page)

if __name__ == "__main__":
    ft.app(target=main)
