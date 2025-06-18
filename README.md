# Name Analysis

## What is it?
A simple tool to visually explore US naming trends over the past ~150 years.

## Where does the data come from?

When you first run the tool, it downloads a zip file from the Social Services Administration. That zip contains data on US names assigned at birth going back to 1880.

## How do I use it?

### Prerequisites

You'll need a Python3 installation, version 3.8 or greater is recommended.

Next you'll need to install the dependencies. The simplest way to do this is with `uv`.
If you have `uv` installed, it will install the dependencies automatically.

### With `uv`

Just navigate to where you downloaded the tool, and execute the command `uv run ./run_gui.py` or `uv run ./run_tui.py`.

### Without `uv`

You'll need to install the dependencies manually with `pip`.

For the TUI version:

```
pip install pandas requests textual textual-plotext python-levenshtein jellyfish 
```

For the GUI version:

```
pip install pandas requests python-levenshtein jellyfish flet matplotlib seaborn flet-desktop
```

Then you can run the command `./run_tui.py` or `./run_gui.py`.

## TUI? GUI?? What???

Name Analysis has two available user interfaces. The TUI, or Textual User Interface runs in the terminal. The GUI, or Graphical User Interface runs a normal independent interface.

### Which one should I use?

It's up to you! They both contain exactly the same feature set. The graphs are prettier in the GUI though :D
