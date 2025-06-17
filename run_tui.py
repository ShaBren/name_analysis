#!/usr/bin/env python

from name_analysis.tui import NameAnalysisApp

if __name__ == "__main__":
    """
    This script is the entry point for running the Textual User Interface (TUI).
    It imports the main application class from the tui module and runs it.
    """
    app = NameAnalysisApp()
    app.run()
