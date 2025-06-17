#!/usr/bin/env python

import pandas as pd
from pathlib import Path
import zipfile
import requests
import io
from typing import Optional, Tuple, Dict, Any
from functools import reduce
import Levenshtein
import jellyfish

# --- Configuration ---
DATA_URL = "https://www.ssa.gov/oact/babynames/names.zip"
DATA_DIR = Path("baby_names_data")
START_YEAR = 1880
END_YEAR = 2023

class DataAnalyzer:
    """Handles all data fetching, loading, and processing logic."""
    def __init__(self, data_dir: Path = DATA_DIR):
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
