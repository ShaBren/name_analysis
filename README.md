<!-- TOC start -->

- [Name Analysis](#name-analysis)
   * [What is it?](#what-is-it)
   * [Where does the data come from?](#where-does-the-data-come-from)
   * [How do I use it?](#how-do-i-use-it)
      + [Prerequisites](#prerequisites)
      + [With `uv`](#with-uv)
      + [Without `uv`](#without-uv)
   * [TUI? GUI?? What???](#tui-gui-what)
      + [Which one should I use?](#which-one-should-i-use)

<!-- TOC end -->

<!-- TOC --><a name="name-analysis"></a>
# Name Analysis

<!-- TOC --><a name="what-is-it"></a>
## What is it?
A simple tool to visually explore US naming trends over the past ~150 years.

<!-- TOC --><a name="where-does-the-data-come-from"></a>
## Where does the data come from?

When you first run the tool, it downloads a zip file from the Social Services Administration. That zip contains data on US names assigned at birth going back to 1880.

<!-- TOC --><a name="how-do-i-use-it"></a>
## How do I use it?

<!-- TOC --><a name="prerequisites"></a>
### Prerequisites

You'll need a Python3 installation, version 3.8 or greater is recommended.

Next you'll need to install the dependencies. The simplest way to do this is with `uv`.
If you have `uv` installed, it will install the dependencies automatically.

<!-- TOC --><a name="with-uv"></a>
### With `uv`

Just navigate to where you downloaded the tool, and execute the command `uv run ./run_gui.py` or `uv run ./run_tui.py`.

<!-- TOC --><a name="without-uv"></a>
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

<!-- TOC --><a name="tui-gui-what"></a>
## TUI? GUI?? What???

Name Analysis has two available user interfaces. The TUI, or Textual User Interface runs in the terminal. The GUI, or Graphical User Interface runs a normal independent interface.

<!-- TOC --><a name="which-one-should-i-use"></a>
### Which one should I use?

It's up to you! They both contain exactly the same feature set. The graphs are prettier in the GUI though :D
