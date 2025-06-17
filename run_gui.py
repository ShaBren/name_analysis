#!/usr/bin/env python

import flet as ft
from name_analysis.gui import main as gui_main

if __name__ == "__main__":
    """
    This script is the entry point for running the Flet-based Graphical User Interface (GUI).
    """
    ft.app(target=gui_main)
