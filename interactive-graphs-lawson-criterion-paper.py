# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Interactive Web-Graphs
# Monarch Kumar

# %%
# %matplotlib inline
# Minor issue - matplotlib is unable to sorce Arial or Helvetica on my device for some reason.
# It still works but throws warning errors, which I have muted here
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# %% [markdown]
# The following contains only relevent and mostly unedited section from the original codebase. The intention is to use original code for calculations and use those variables as-is for web-enabled graphs.
# <HR><HR>

# %% [markdown]
# # Configuration Setup

# %% jupyter={"source_hidden": true}
import math
import sys
import os
import configparser
from decimal import Decimal

import scipy
from scipy import integrate
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib import ticker
from matplotlib.ticker import StrMethodFormatter, NullFormatter
from tqdm import tqdm
from PIL import Image
import glob


# Import our library/utility fuctions
from lib import latexutils
from lib import fusionlib
from lib import cross_section
from lib import reactivity
from lib import conversions
from lib import plasmaprofile
from lib import experiment
from lib import exceptions

# Plot styles
plt.style.use(['./styles/medium.mplstyle'])

# DPI and figure sizes for paper
dpi = 300
figsize = (3,3)
figsize_fullpage = (8,8)

# Setup plots to use LaTeX
latex_params = { 
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{siunitx} \usepackage{mathtools}",
}
mpl.rcParams.update(latex_params)

# Color choices
four_colors = ['red', 'purple', 'slateblue', 'navy']
two_colors = ['red', 'navy']
reaction_color_dict = {'T(d,n)4He': 'blue',
                       'D(d,p)T': 'lightgreen',
                       'D(d,n)3He': 'darkgreen',
                       'CATDD': 'green',
                       '3He(d,p)4He': 'red',
                       '11B(p,4He)4He4He': 'purple',
                      }

# Ploting options
add_prepublication_watermark = True

# Naming for figures
label_filename_dict = {
    ### Figures
    #'fig:E_f_trinity': 'fig_1.png', # Illustration
    'fig:scatterplot_ntauE_vs_T': 'fig_2.png',
    'fig:scatterplot_nTtauE_vs_year': 'fig_3.png',
    'fig:reactivities': 'fig_4.png',
    #'fig:lawsons_1st': 'fig_5.png', # Illustration
    'fig:ideal_ignition': 'fig_6.png',
    #'fig:power_balances': 'fig_6.png', # Illustration
    #'fig:lawsons_2nd': 'fig_7.png', # Illustration
    'fig:Q_vs_T': 'fig_8.png',
    #'fig:lawsons_generalized': 'fig_9.png' # Illustration
    'fig:Q_vs_T_extended': 'fig_10.png',
    #'fig:power_balances': 'fig_11.png', # Illustration
    'fig:MCF_ntau_contours_q_fuel_q_sci': 'fig_12.png',
    #'fig:conceptual_icf_basic': 'fig_13.png' # Illustration
    'fig:ICF_ntau_contours_q_fuel_q_sci': 'fig_14.png',
    'fig:MCF_nTtau_contours_q_fuel': 'fig_15.png',
    'fig:scatterplot_nTtauE_vs_T': 'fig_16.png',
    #'fig:conceptual_plant': 'fig_16.png' # Illustration
    'fig:Qeng': 'fig_18.png',
    'fig:Qeng_high_efficiency': 'fig_19.png',
    'fig:parabolic_profiles_a': 'fig_20a.png',
    'fig:parabolic_profiles_b': 'fig_20b.png',
    'fig:parabolic_profiles_c': 'fig_20c.png',
    'fig:peaked_broad_profiles_a': 'fig_21a.png',
    'fig:peaked_broad_profiles_b': 'fig_21b.png',
    'fig:peaked_broad_profiles_c': 'fig_21c.png',
    'fig:nTtauE_vs_T_peaked_and_broad_bands': 'fig_22.png',
    #'fig:conceptual_icf_detailed': 'fig_23.png' # Illustration
    #'fig:z_pinch': 'fig_24.png' # Illustration
    'fig:bennett_profiles': 'fig_25.png',
    'fig:effect_of_bremsstrahlung_a': 'fig_26a.png',
    'fig:effect_of_bremsstrahlung_b': 'fig_26b.png',
    'fig:D-3He_a': 'fig_27a.png',
    'fig:D-3He_b': 'fig_27b.png',
    'fig:pB11_vs_bremsstrahlung': 'fig_28.png',
    'fig:pB11_a': 'fig_29a.png',
    'fig:pB11_b': 'fig_29b.png',
    'fig:CAT_D-D_a': 'fig_30a.png',
    'fig:CAT_D-D_b': 'fig_30b.png',
    'fig:all_reactions_a': 'fig_31a.png',
    'fig:all_reactions_b': 'fig_31b.png',
    #'fig:conceptual_plant_non_electrical_recirculating': 'fig_33.png' # Illustration
    'fig:Qeng_appendix': 'fig_33.png',
    #'fig:torus_cross_section': 'fig_34.png', # Illustration
    ### Tables
    'tab:glossary': 'table_1.tex',
    'tab:minimum_lawson_parameter_table': 'table_2.tex',
    'tab:minimum_triple_product_table': 'table_3.tex',
    'tab:efficiency_table': 'table_4.tex',
    'tab:mcf_peaking_values_table': 'table_5.tex',
    'tab:mainstream_mcf_data_table': 'table_6.tex',
    'tab:alternates_mcf_data_table': 'table_7.tex',
    'tab:icf_mif_data_table': 'table_8.tex',
    'tab:q_sci_data_table': 'table_9.tex',
}

# Initialize configparser
config = configparser.ConfigParser()

# Uncomment below to show all columns when printing dataframes
pd.set_option('display.max_columns', None)
# Uncomment below to show all rows when printing dataframes
#pd.set_option('display.max_rows', None)

# Create required folders
if not os.path.exists('tables_latex'):
    os.makedirs('tables_latex')
if not os.path.exists('tables_csv'):
    os.makedirs('tables_csv')
if not os.path.exists('images'):
    os.makedirs('images')
if not os.path.exists('animation'):
    os.makedirs('animation')

print('Setup complete.')

# %% [markdown] hidden=true
# # Data Plots
#
# This section handles the generation of the plots which contain experimental datapoints (and also plots of the countors of Q for various profile and impurity effects).

# %% [markdown]
# ## Calculate DT requirements accounting for adjustments (profiles, impurities, $C_B$)

# %% hidden=true jupyter={"source_hidden": true}
# number_of_temperature_values sets the number of temperature values for all further plots
number_of_temperature_values = 300
log_temperature_values = np.logspace(math.log10(0.5), math.log10(100), number_of_temperature_values)

# Initialize the dataframe with the temperature values
DT_requirements_df = pd.DataFrame(log_temperature_values, columns=['T_i0'])

# Define the Q values to be evaluated (and eventually plotted)
Qs = [float('inf'), 10, 2, 1, 0.1, 0.01, 0.001]

# Define the experiments to be evaluated
# see lib/experiment.py for the definitions of the experiment classes
experiments = [experiment.UniformProfileDTExperiment(),
               experiment.UniformProfileHalfBremsstrahlungDTExperiment(),
               experiment.LowImpurityPeakedAndBroadDTExperiment(),
               experiment.HighImpurityPeakedAndBroadDTExperiment(),
               experiment.IndirectDriveICFDTExperiment(),
               experiment.IndirectDriveICFDTBettiCorrectionExperiment(),
               experiment.ParabolicProfileDTExperiment(),
               experiment.PeakedAndBroadDTExperiment(),
              ]

# Initialize a dictionary to hold all the new columns
new_columns = {}

# note that hipabdt stands for high impurity peaked and broad deuterium tritium
# note that lipabdt stands for low impurity peaked and broad deuterium tritium
# note that pabdt stands for peaked and broad deuterium tritium
# Run the calculations for each experiment and Q value. This is a bit slow.
for ex in experiments:
    print(f'Calculating lawson parameter and triple product requirements for {ex.name}...')
    for Q in Qs:
        # Calculate triple product needed to achieve Q_fuel
        new_columns[ex.name + '__nTtauE_Q_fuel=' + str(Q)] = DT_requirements_df['T_i0'].apply(
            lambda T_i0: ex.triple_product_Q_fuel(T_i0=T_i0, Q_fuel=Q)
        )
        # Calculate Lawson parameter needed to achieve Q_fuel
        new_columns[ex.name + '__ntauE_Q_fuel=' + str(Q)] = DT_requirements_df['T_i0'].apply(
            lambda T_i0: ex.lawson_parameter_Q_fuel(T_i0=T_i0, Q_fuel=Q)
        )
        # Calculate triple product needed to achieve Q_sci
        new_columns[ex.name + '__nTtauE_Q_sci=' + str(Q)] = DT_requirements_df['T_i0'].apply(
            lambda T_i0: ex.triple_product_Q_sci(T_i0=T_i0, Q_sci=Q)
        )
        # Calculate Lawson parameter needed to achieve Q_sci
        new_columns[ex.name + '__ntauE_Q_sci=' + str(Q)] = DT_requirements_df['T_i0'].apply(
            lambda T_i0: ex.lawson_parameter_Q_sci(T_i0=T_i0, Q_sci=Q)
        )
print("Calculations complete. Converting to dataframe...")
# Convert the dictionary to a DataFrame
new_columns_df = pd.DataFrame(new_columns)

# Concatenate the new columns with the original DataFrame
DT_requirements_df = pd.concat([DT_requirements_df, new_columns_df], axis=1)

# Required for obtaining clean looking plots
# When plotting later, in order for ax.fill_between to correctly fill the region that goes to
# infinity, the values of infinity in the dataframe must be replaced with non-infinite values.
# We replace the infinities with 1e30 here, far beyond the y limit of any plots.
DT_requirements_df = DT_requirements_df.replace(math.inf, 1e30)

print("Done.")

# %% [markdown]
# ## Calculate Lawson parameter, triple product, and p-tau minima

# %% hidden=true jupyter={"source_hidden": true}
# Evaluate minimum triple products and Lawson parameters to achieve various levels of Q.
# Used to create bands on triple product vs time graph.

data = {'requirement':[], 'minimum_value':[], 'T_i0':[]}
for col in DT_requirements_df.columns:
    if col != 'T_i0':
        i = DT_requirements_df[col].idxmin()
        ion_temperature = DT_requirements_df.iloc[i]['T_i0']
        minimum_value = DT_requirements_df.iloc[i][col]
        
        data['requirement'].append(col)
        data['T_i0'].append(ion_temperature)
        data['minimum_value'].append(minimum_value)

DT_requirement_minimum_values_df = pd.DataFrame(data)

conditions_of_interest = ['uniform_profile_experiment__nTtauE_Q_sci=inf',
                         'uniform_profile_experiment__nTtauE_Q_sci=1',
                         'hipabdt_experiment__nTtauE_Q_sci=inf', # MCF upper bound
                         'lipabdt_experiment__nTtauE_Q_sci=inf', # MCF lower bound
                         'hipabdt_experiment__nTtauE_Q_sci=1', # MCF upper bound
                         'lipabdt_experiment__nTtauE_Q_sci=1', # MCF lower bound
                         ]
# Print out the minimum triple product and temperature for each Q requirement to show
# how the peak temperature requirements increase with profiles.
for i, req in enumerate(DT_requirement_minimum_values_df['requirement']):
    if req in conditions_of_interest:
        print(f"The minimum value of {req} is {DT_requirement_minimum_values_df.iloc[i]['minimum_value']:.2e} at T_i0 = {DT_requirement_minimum_values_df.iloc[i]['T_i0']:.1f} keV")


# %% [markdown]
# ## Analysis of Experimental Results

# %% [markdown]
# ### Load experimental data

# %% jupyter={"source_hidden": true}
# Get the raw experimental result dataframe
filename = 'data/experimental_results.pkl'
experimental_result_df = pd.read_pickle(filename)
# Note the temperatures are stored as strings with the approprate
# number of significant figures so no changes occur here.

# Convert scientific notation strings to floats
experimental_result_df['n_e_avg'] = experimental_result_df['n_e_avg'].astype(float)
experimental_result_df['n_e_max'] = experimental_result_df['n_e_max'].astype(float)
experimental_result_df['n_i_avg'] = experimental_result_df['n_i_avg'].astype(float)
experimental_result_df['n_i_max'] = experimental_result_df['n_i_max'].astype(float)
experimental_result_df['tau_E'] = experimental_result_df['tau_E'].astype(float)
experimental_result_df['tau_E_star'] = experimental_result_df['tau_E_star'].astype(float)
experimental_result_df['Z_eff'] = experimental_result_df['Z_eff'].astype(float)
experimental_result_df['rhoR_tot'] = experimental_result_df['rhoR_tot'].astype(float)
experimental_result_df['YOC'] = experimental_result_df['YOC'].astype(float)
experimental_result_df['p_stag'] = experimental_result_df['p_stag']
experimental_result_df['tau_stag'] = experimental_result_df['tau_stag'].astype(float)

experimental_result_df['E_ext'] = experimental_result_df['E_ext'].astype(float)
experimental_result_df['E_F'] = experimental_result_df['E_F'].astype(float)
experimental_result_df['P_ext'] = experimental_result_df['P_ext'].astype(float)
experimental_result_df['P_F'] = experimental_result_df['P_F'].astype(float)

# DATE HANDLING
# If the date field exists, clear the year field so we know to use the full date
experimental_result_df['Year'] = experimental_result_df['Year'].mask(experimental_result_df['Date'].notna(), None)

# Convert Date field to datetime, falling back to January 1st of Year field if Date is missing
experimental_result_df['Date'] = pd.to_datetime(experimental_result_df['Date']).fillna(
    pd.to_datetime(experimental_result_df[experimental_result_df['Year'].notna()]['Year'].astype(int).astype(str) + '-01-01')
)

# If the Year field is None, use the Date field. Otherwise use the Year field
for row in experimental_result_df.itertuples():
    if pd.isnull(row.Year):
        experimental_result_df.at[row.Index, 'Display Date'] = row.Date.strftime('%Y-%m-%d')
    else:
        experimental_result_df.at[row.Index, 'Display Date'] = str(int(row.Year))

# Sort by Display Date upfront here so that the downstream latex tables are in date order
experimental_result_df = experimental_result_df.sort_values(by='Display Date')

# For updated paper, to keep the references short, refer to our 2022 paper for unchanged data
mask = experimental_result_df['new_or_changed_2025_update'] == False
experimental_result_df.loc[mask, 'Bibtex Strings'] = experimental_result_df.loc[mask, 'Bibtex Strings'].apply(lambda x: [r'2022_Wurzel_Hsu'])


# %% [markdown]
# ### Split experimental results into separate Q_sci, MCF, and MIF/ICF dataframes, define headers for dataframe and latex tables. 

# %% jupyter={"source_hidden": true}
#######################
# Q_sci
Q_sci_experimental_result_df = experimental_result_df.loc[experimental_result_df['include_Qsci_vs_date_plot']]
q_sci_airtable_latex_map = {
    'Project Displayname': 'Project',
    'Concept Displayname': 'Concept',
    'Display Date': 'Date',
    'Shot': 'Shot identifier',
    'Bibtex Strings': 'Reference',
    'E_ext': r'\thead{$E_{\rm in}$ \\ (\si{J})}',
    'E_F': r'\thead{$Y$ \\ (\si{J})}',
    'P_ext': r'\thead{$P_{\rm in}$ \\ (\si{W})}',
    'P_F': r'\thead{$P_{\rm F}$ \\ (\si{W})}',
}

q_sci_calculated_latex_map = {
    'Q_sci': r'\thead{$Q_{\rm sci}$ \\ }',
}

q_sci_keys = list(q_sci_airtable_latex_map.keys()) + ['Date']
q_sci_df = Q_sci_experimental_result_df.filter(items=q_sci_keys)

#######################
# MCF
MCF_concepts = ['Tokamak', 'Spherical Tokamak', 'Stellarator', 'RFP', 'Pinch', 'Spheromak', 'Mirror', 'Z Pinch', 'FRC', 'MTF']
mcf_experimental_result_df = experimental_result_df.loc[
    (experimental_result_df['Concept Displayname'].isin(MCF_concepts)) & 
    (experimental_result_df['include_lawson_plots'] == True)
]

# Mapping from data column headers to what should be printed in latex tables
mcf_airtable_latex_map = {
    'Project Displayname': 'Project',
    'Concept Displayname': 'Concept',
    'Display Date': 'Date',
    'Shot': 'Shot identifier',
    'Bibtex Strings': 'Reference',
    'T_i_max': r'\thead{$T_{i0}$ \\ (\si{keV})}',
    'T_i_avg': r'\thead{$\langle T_{i} \rangle$ \\ (\si{keV})}',
    'T_e_max': r'\thead{$T_{e0}$ \\ (\si{keV})}',
    'T_e_avg': r'\thead{$\langle T_{e} \rangle$ \\ (\si{keV})}',
    'n_i_max': r'\thead{$n_{i0}$ \\ (\si{m^{-3}})}',
    'n_i_avg': r'\thead{$\langle n_{i} \rangle$ \\ (\si{m^{-3}})}',
    'n_e_max': r'\thead{$n_{e0}$ \\ (\si{m^{-3}})}',
    'n_e_avg': r'\thead{$\langle n_{e} \rangle$ \\ (\si{m^{-3}})}',
    'Z_eff': r'$\thead{Z_{eff} \\ }$',
    'tau_E_star': r'\thead{$\tau_{E}^{*}$ \\ (\si{s})}',
    'tau_E': r'\thead{$\tau_{E}$ \\ (\si{s})}$',
}

# Mapping from what's calculated in this code to what should be printed in latex tables
mcf_calculated_latex_map = {
    'ntauEstar_max': r'\thead{$n_{i0} \tau_{E}^{*}$ \\ (\si{m^{-3}~s})}',
    'nTtauEstar_max': r'\thead{$n_{i0} T_{i0} \tau_{E}^{*}$ \\ (\si{keV~m^{-3}~s})}',
}

# Only keep columns that are relevant to MCF. Also add the Date column since it's
# not included in the airtable_latex_map but is needed for plots.   
mcf_keys = list(mcf_airtable_latex_map.keys()) + ['Date']
mcf_df = mcf_experimental_result_df.filter(items=mcf_keys)

#######################
# ICF/MIF
ICF_MIF_concepts = ['Laser Direct Drive', 'Laser Indirect Drive', 'MagLIF']
icf_mif_experimental_result_df = experimental_result_df.loc[
    (experimental_result_df['Concept Displayname'].isin(ICF_MIF_concepts)) & 
    (experimental_result_df['include_lawson_plots'] == True)
]

# Mapping from data column headers to what should be printed in latex tables
icf_mif_airtable_latex_map = {
    'Project Displayname': 'Project',
    'Concept Displayname': 'Concept',
    'Display Date': 'Date',
    'Shot': 'Shot identifier',
    'Bibtex Strings': 'Reference',
    'T_i_avg': r'\thead{$\langle T_i \rangle_{\rm n}$ \\ (\si{keV})}',
    'T_e_avg': r'\thead{$T_e$ \\ (\si{keV})}',
    'rhoR_tot': r'\thead{$\rho R_{tot(n)}^{no (\alpha)}$ \\ (\si{g/cm^{-2}})}',
    'YOC': r'YOC',
    'p_stag': r'\thead{$p_{stag}$ \\ (\si{Gbar})}',
    'tau_stag': r'\thead{$\tau_{stag}$ \\ (\si{s})}',
    'E_ext': r'\thead{$E_{\rm in}$ \\ (\si{J})}',
    'E_F': r'\thead{$Y$ \\ (\si{J})}',
    #'P_ext': r'\thead{$P_{\rm in}$ \\ (\si{W})}',
    #'P_F': r'\thead{$P_{\rm F}$ \\ (\si{W})}',
}

# Mapping from what's calculated in this code to what should be printed in latex tables
icf_mif_calculated_latex_map = {
    'ptau': r'\thead{$P\tau_{\rm stag}$ \\ (\si{atm~s})}',
    'ntauE_avg': r'\thead{$n\tau_{\rm stag}$ \\ (\si{m^{-3}~s})}',
    'nTtauE_avg': r'\thead{$n \langle T \rangle_{\rm n} \tau_{\rm stag}$ \\ (\si{keV~m^{-3}~s})}',

}
# Only keep columns that are relevant to ICF/MIF. Also add the Date column since it's
# not included in the airtable_latex_map.   
icf_mif_keys = list(icf_mif_airtable_latex_map.keys()) + ['Date']
icf_mif_df = icf_mif_experimental_result_df.filter(items=icf_mif_keys)
icf_mif_df
print(f'Split data into {len(q_sci_df)} Q_sci experimental results, {len(mcf_df)} MCF experimental results and {len(icf_mif_df)} MIF/ICF results.')

# %% [markdown]
# ### Calculate Q_sci values

# %% jupyter={"source_hidden": true}
# Q_sci is calculated from E_ext and E_F or P_ext and P_F
# Calculate Q_sci using either energy or power ratios
q_sci_df['Q_sci'] = q_sci_df['E_F'] / q_sci_df['E_ext']  # try energy first
mask = q_sci_df['Q_sci'].isna()  # where energy calculation failed
q_sci_df.loc[mask, 'Q_sci'] = q_sci_df.loc[mask, 'P_F'] / q_sci_df.loc[mask, 'P_ext']  # try power instead
#q_sci_df

# %% [markdown]
# ### Make LaTeX dataframe for Q_sci experimental data, save data tables

# %% jupyter={"source_hidden": true}
# Note we are relying on ordered dictionaries here so the headers keys (dataframe headers)
# line up correctly with the table header values (latex headers)
# Ordered dictionaries are a feature of Python 3.7+ See this link for more info:
# https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6

# header keys are the dataframe headers
header_keys = {**q_sci_airtable_latex_map, **q_sci_calculated_latex_map}.keys()
# Set final order of columns in new table
latex_q_sci_df = q_sci_df[header_keys]

def format_q_sci_experimental_result(row):
    if not math.isnan(row['E_ext']):
        row['E_ext'] = '{:.1e}'.format(row['E_ext'])
        row['E_ext'] = latexutils.siunitx_num(row['E_ext'])
    if not math.isnan(row['E_F']):
        row['E_F'] = '{:.1e}'.format(row['E_F'])
        row['E_F'] = latexutils.siunitx_num(row['E_F'])
    if not math.isnan(row['P_ext']):
        row['P_ext'] = '{:.1e}'.format(row['P_ext'])
        row['P_ext'] = latexutils.siunitx_num(row['P_ext'])
    if not math.isnan(row['P_F']):
        row['P_F'] = '{:.1e}'.format(row['P_F'])
        row['P_F'] = latexutils.siunitx_num(row['P_F'])

    row['Q_sci'] = '{:.2f}'.format(row['Q_sci'])
    
    row['Bibtex Strings'] = latexutils.cite(row['Bibtex Strings'])    
    return row

# Format values
latex_q_sci_df = latex_q_sci_df.apply(lambda row: format_q_sci_experimental_result(row), axis=1)
# Rename column headers
latex_q_sci_df = latex_q_sci_df.rename(columns={**q_sci_airtable_latex_map, **q_sci_calculated_latex_map})    

caption = "Data for experiments which produced sufficient fusion energy to achieve appreciable values of scientific gain $Q_{\mathrm{sci}}$."
label = "tab:q_sci_data_table"

latexutils.latex_table_to_csv(latex_q_sci_df, "tables_csv/q_sci_data.csv")

q_sci_table_latex = latex_q_sci_df.to_latex(
                         caption=caption,
                         label=label,
                         escape=False,
                         na_rep=latexutils.table_placeholder,
                         index=False,
                         formatters={},
                      )
# Post processing of latex code to display as desired
q_sci_table_latex = latexutils.JFE_comply(q_sci_table_latex)
q_sci_table_latex = latexutils.full_width_table(q_sci_table_latex)
q_sci_table_latex = latexutils.sideways_table(q_sci_table_latex)

fh=open(os.path.join('tables_latex', label_filename_dict[label]), 'w')
fh.write(q_sci_table_latex)
fh.close()


# %% [markdown] heading_collapsed=true
# ### Infer, and calculate ICF and MIF values

# %% hidden=true jupyter={"source_hidden": true}
def ptau_betti_2010(rhoR_tot, T_i_avg, YOC, mu=0.5):
    """Calculate the effective ptau using Betti's 2010 approach and return
    pressure * confinement time in atm s.
    
    The details of this approach are published here,
    https://doi.org/10.1063/1.3380857
    This approach is limited to sub-ignited capsules that don't produce
    signficant alpha heating. In practice it's only used for older OMEGA
    shots where pressure is not inferred.
    From private communication with Betti, this approach should not
    be used with more recent data as now pressure is inferred and is reported
    directly. See function ptau_betti_2019.
    
    Keyword arguments:
    rhoR_tot -- total areal densityin g/cm^2
    T_i_avg -- average ion temperature over burn in keV
    YOC -- yield over clean
    mu -- mu as defined in https://doi.org/10.1063/1.3380857
          mu is fixed at 0.5 as suggested in this paper Section IV.B.1
    """
    ptau = 8 * ((rhoR_tot*float(T_i_avg))**0.8) * (YOC**mu)
    return ptau

def ptau_betti_2019(p_stag_Gbar, tau_burn_s):
    """
    THIS FUNCTION IS DEPRECATED. Use ptau_direct instead. See 2025 paper for details.

    Calculate the effective ptau using Betti's 2019 approach and return
    pressure * confinement time in atm s.
    
    This approach takes inferred pressure p_stag in Gbar
    According to private communication with Betti ptau should be calculated
    directly rather than using his 2010 paper approach. However a correction
    factor of 0.93/(2*1.4) times the burn time is needed to get a confinement
    time for which ignition corresponds to the onset of propagating burn.
    This is from private communication and is also published here,
    https://doi.org/10.1103/PhysRevE.99.021201
    
    Keyword arguments:
    p_stag_Gbar -- inferred stagnation pressure in Gbar
    tau_burn_s -- burn duration in s. FWHM of neutron emissions.
    """
    raise ValueError('This function is deprecated. Use ptau_direct instead.')
    # First convert p_stag from Gbar to atm
    p_stag_atm = p_stag_Gbar * conversions.atm_per_gbar
    
    # In the original paper, we applied this correction factor here.
    # In the updated paper we apply it to the NIF ignition contour only.
    # We approximate the confinement time tau as tau_burn * [0.93/(2*1.4)]
    # per https://doi.org/10.1103/PhysRevE.99.021201
    ### May need to change 2 ---> 4, See Atzeni p. 40
    ### #atzeni_betti_factor = 0.93/(4*1.4)
    betti_factor = 0.93/(2*1.4)
    tau = tau_burn_s * betti_factor
    
    ptau = p_stag_atm * tau
    return ptau

def ptau_direct(p_stag_Gbar, tau_burn_s):
    """Directly an effective ptau value with no corrections.
    
    Here we assume tau_burn is the confinement time exactly.
    
    Keyword arguments:
    p_stag_Gbar -- Inferred stagnation pressure in Gbar
    tau_burn_s -- Burn time in seconds aka tau_stag
    """
    # First convert p_stag from Gbar to atm
    p_stag_atm = p_stag_Gbar * conversions.atm_per_gbar
    ptau = p_stag_atm * tau_burn_s
    return ptau
    
def icf_mif_calculate(row):
    """Calculate ptau and nTtau_E for ICF and MIF experiments.
    
    The approach for calculating ptau varies. See paper for details.
    """
    # Use Betti 2010 for older OMEGA shots without reported pressure
    if row['Project Displayname'] == 'OMEGA' and pd.isnull(row['p_stag']):   
        row['ptau'] = ptau_betti_2010(rhoR_tot=row['rhoR_tot'],
                                      T_i_avg=row['T_i_avg'],
                                      YOC=row['YOC'])
    elif row['Project Displayname'] == 'NOVA':
        row['ptau'] = ptau_direct(p_stag_Gbar=float(row['p_stag']),
                                      tau_burn_s=row['tau_stag']) 
    elif row['Project Displayname'] == 'OMEGA' and not pd.isnull(float(row['p_stag'])):
        row['ptau'] = ptau_direct(p_stag_Gbar=float(row['p_stag']),
                                      tau_burn_s=row['tau_stag'])
    elif row['Project Displayname'] == 'NIF':
        row['ptau'] = ptau_direct(p_stag_Gbar=float(row['p_stag']),
                                      tau_burn_s=row['tau_stag'])   
    elif row['Project Displayname'] == 'MagLIF':
        row['ptau'] = ptau_direct(p_stag_Gbar=float(row['p_stag']),
                                  tau_burn_s=row['tau_stag'])
    elif row['Project Displayname'] == 'FIREX':
        row['ptau'] = ptau_direct(p_stag_Gbar=float(row['p_stag']),
                                  tau_burn_s=row['tau_stag'])
    else:
        raise ValueError(f'''Could not find a method for calculating ptau for
                           {row['Project Displayname']}. Stopping.''')
    
    # Once ptau is calculated, calculating nTtau_E is the same
    row['nTtauE_avg'] = conversions.ptau_to_nTtau_E(row['ptau'])
    # ntau_E is obtained simply by dividing out the ion temp (except for FIREX)
    if row['Project Displayname'] == 'FIREX':
        row['ntauE_avg'] = row['nTtauE_avg'] / float(row['T_e_avg'])
    else:
        row['ntauE_avg'] = row['nTtauE_avg'] / float(row['T_i_avg'])
    return row

icf_mif_df = icf_mif_df.apply(lambda row: icf_mif_calculate(row), axis=1)
#icf_mif_df

# %% [markdown] heading_collapsed=true
# ### Make LaTeX dataframe for ICF/MIF experimental data, save data tables

# %% hidden=true jupyter={"source_hidden": true}
# Note we are relying on ordered dictionaries here so the headers keys (dataframe headers)
# line up correctly with the table header values (latex headers)
# Ordered dictionaries are a feature of Python 3.7+ See this link for more info:
# https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6

# header keys are the dataframe headers
header_keys = {**icf_mif_airtable_latex_map, **icf_mif_calculated_latex_map}.keys()
# header values are the corresponding latex table headers
header_values = {**icf_mif_airtable_latex_map, **icf_mif_calculated_latex_map}.values()
# Set final order of columns in new table
latex_icf_mif_df = icf_mif_df[header_keys]

def format_icf_mif_experimental_result(row):
    if not math.isnan(row['ptau']):
        row['ptau'] = '{:.2f}'.format(row['ptau'])
        row['ptau'] = latexutils.siunitx_num(row['ptau'])

    if not math.isnan(row['rhoR_tot']):
        row['rhoR_tot'] = '{:.3f}'.format(row['rhoR_tot'])

    if not math.isnan(row['YOC']):
        row['YOC'] = '{:.1f}'.format(row['YOC'])

    if not math.isnan(row['E_ext']):
        row['E_ext'] = '{:.1e}'.format(row['E_ext'])
        row['E_ext'] = latexutils.siunitx_num(row['E_ext'])
    if not math.isnan(row['E_F']):
        row['E_F'] = '{:.1e}'.format(row['E_F'])
        row['E_F'] = latexutils.siunitx_num(row['E_F'])

    row['tau_stag'] = latexutils.siunitx_num(row['tau_stag'])
    
    row['nTtauE_avg'] = '{:0.1e}'.format(row['nTtauE_avg'])
    row['nTtauE_avg'] = latexutils.siunitx_num(row['nTtauE_avg'])
    
    row['ntauE_avg'] = '{:0.1e}'.format(row['ntauE_avg'])
    row['ntauE_avg'] = latexutils.siunitx_num(row['ntauE_avg'])
      
    row['Bibtex Strings'] = latexutils.cite(row['Bibtex Strings'])    
    return row

# Format values
latex_icf_mif_df = latex_icf_mif_df.apply(lambda row: format_icf_mif_experimental_result(row), axis=1)
# Rename column headers
latex_icf_mif_df = latex_icf_mif_df.rename(columns={**icf_mif_airtable_latex_map, **icf_mif_calculated_latex_map})    

caption = "Data for ICF and higher-density MIF concepts."
label = "tab:icf_mif_data_table"

latexutils.latex_table_to_csv(latex_icf_mif_df, "tables_csv/icf_mif_data.csv")

icf_mif_table_latex = latex_icf_mif_df.to_latex(
                         caption=caption,
                         label=label,
                         escape=False,
                         na_rep=latexutils.table_placeholder,
                         index=False,
                         formatters={},
                      )
# Post processing of latex code to display as desired
icf_mif_table_latex = latexutils.JFE_comply(icf_mif_table_latex)
icf_mif_table_latex = latexutils.full_width_table(icf_mif_table_latex)
icf_mif_table_latex = latexutils.sideways_table(icf_mif_table_latex)

fh=open(os.path.join('tables_latex', label_filename_dict[label]), 'w')
fh.write(icf_mif_table_latex)
fh.close()

# %% [markdown]
# ### Set peaking values for MCF 

# %% jupyter={"source_hidden": true}
# Values of peaking depend on Concept Type. Some are calculated from
# profiles, some are quoted directly.
spherical_tokamak_profile = plasmaprofile.SphericalTokamakProfile()
spherical_tokamak_peaking_temperature = spherical_tokamak_profile.peaking_temperature()
spherical_tokamak_peaking_density = spherical_tokamak_profile.peaking_density()

tokamak_profile = plasmaprofile.TokamakProfile()
tokamak_peaking_temperature = tokamak_profile.peaking_temperature()
tokamak_peaking_density = tokamak_profile.peaking_density()

stellarator_profile = plasmaprofile.StellaratorProfile()
stellarator_peaking_temperature = stellarator_profile.peaking_temperature()
stellarator_peaking_density = stellarator_profile.peaking_density()

frc_peaking_temperature = 1
frc_peaking_density = 1.3
frc_citations = ['Slough_1995', 'Steinhauer_2018']

rfp_peaking_temperature = 1.2
rfp_peaking_density = 1.2
rfp_citations = ['Chapman_2002']


peaking_dict = {'Tokamak': {'peaking_temperature': tokamak_peaking_temperature,
                            'peaking_density': tokamak_peaking_density,
                            'citations': tokamak_profile.citations},
                'Stellarator': {'peaking_temperature': stellarator_peaking_temperature,
                                'peaking_density': stellarator_peaking_density,
                                'citations': stellarator_profile.citations},
                'Spherical Tokamak': {'peaking_temperature': spherical_tokamak_peaking_temperature,
                                      'peaking_density': spherical_tokamak_peaking_density,
                                      'citations': spherical_tokamak_profile.citations},
                'FRC': {'peaking_temperature': frc_peaking_temperature,
                        'peaking_density': frc_peaking_density,
                        'citations': frc_citations},
                'RFP': {'peaking_temperature': rfp_peaking_temperature,
                        'peaking_density': rfp_peaking_density,
                        'citations': rfp_citations},
                'Spheromak': {'peaking_temperature': 2,
                              'peaking_density': 1.5,
                              'citations': ['Hill_2000']},
                # Peaking factors are not needed for the following concepts
                #'Z Pinch': {'peaking_temperature': 2,
                #            'peaking_density': 2},
                #'Pinch': {'peaking_temperature': 2,
                #          'peaking_density': 2},
                #'Mirror': {'peaking_temperature': 2,
                #           'peaking_density': 2},
               }

# %% [markdown]
# ### Make LaTeX table for peaking values

# %% jupyter={"source_hidden": true}
peaking_dict_for_df = {'Concept': list(peaking_dict.keys()),
                       'Peaking Temperature': [peaking_dict.get(concept).get('peaking_temperature') for concept in list(peaking_dict.keys())],
                       'Peaking Density': [peaking_dict.get(concept).get('peaking_density') for concept in list(peaking_dict.keys())],
                       'Reference': [latexutils.cite(peaking_dict.get(concept).get('citations')) for concept in list(peaking_dict.keys())],
                       #'Citation': [' '.join([f'\cite{{{citation}}}' for citation in peaking_dict.get(concept).get('citations', [])]) for concept in list(peaking_dict.keys())]
                      }
peaking_df = pd.DataFrame.from_dict(peaking_dict_for_df)
peaking_df

label='tab:mcf_peaking_values_table'

with pd.option_context("max_colwidth", 1000):
    mcf_peaking_values_table_latex = peaking_df.to_latex(
                      caption=r'Peaking values required to convert reported volume-averaged quantities to peak value quantities.',
                      label=label,
                      escape=False,
                      index=False,
                      formatters={},
                      na_rep=latexutils.table_placeholder,
                      header=['Concept', r'$T_0 / \langle T \rangle$', r'$n_0 / \langle n \rangle$', 'Reference']
                      )
    mcf_peaking_values_table_latex = latexutils.JFE_comply(mcf_peaking_values_table_latex)
    #mcf_table_latex = latexutils.include_table_footnote(mcf_peaking_values_table_latex, 'some footnote')
    #print(mcf_peaking_values_table_latex)
    fh=open(os.path.join('tables_latex', label_filename_dict[label]), 'w')
    fh.write(mcf_peaking_values_table_latex)
    fh.close()
peaking_df


# %% [markdown]
# ### Adjust, infer, and calculate MCF values

# %% jupyter={"source_hidden": true}
def process_mcf_experimental_result(row):
    ### Adjust peaking values based on 
    peaking_temperature = peaking_dict.get(row['Concept Displayname'], {}).get('peaking_temperature', None)
    peaking_density = peaking_dict.get(row['Concept Displayname'], {}).get('peaking_density', None)
    
    ### Set all inferred flags to false initially
    row['inferred_T_i_max_from_T_e_max'] = False
    row['inferred_T_i_max_from_T_i_avg'] = False
    row['inferred_T_i_max_from_T_e_avg'] = False

    row['inferred_n_i_max_from_n_e_max'] = False
    row['inferred_n_i_max_from_n_i_avg'] = False    
    row['inferred_n_i_max_from_n_e_avg'] = False

    row['inferred_tau_E_star_from_tau_E'] = False

    ### Infer missing ion temperatures if necessary###
    if pd.isnull(row['T_i_max']) and not pd.isnull(row['T_i_avg']):
        row['T_i_max'] = float(row['T_i_avg']) * peaking_temperature
        #TODO sigfigs
        row['inferred_T_i_max_from_T_i_avg'] = True
    elif pd.isnull(row['T_i_max']) and not pd.isnull(row['T_e_max']):
        row['T_i_max'] = row['T_e_max']
        row['inferred_T_i_max_from_T_e_max'] = True
    elif pd.isnull(row['T_i_max']) and not pd.isnull(row['T_e_avg']):
        row['T_i_max'] = row['T_e_avg'] * peaking_temperature
        row['inferred_T_i_max_from_T_e_avg'] = True    

    ### Infer missing ion densities if necessary###
    if pd.isnull(row['n_i_max']) and not pd.isnull(row['n_i_avg']):
        row['n_i_max'] = row['n_i_avg'] * peaking_density
        row['inferred_n_i_max_from_n_i_avg'] = True
    elif pd.isnull(row['n_i_max']) and not pd.isnull(row['n_e_max']):
        row['n_i_max'] = row['n_e_max']
        row['inferred_n_i_max_from_n_e_max'] = True
    elif pd.isnull(row['n_i_max']) and not pd.isnull(row['n_e_avg']):
        row['n_i_max'] = row['n_e_avg'] * peaking_density
        row['inferred_n_i_max_from_n_e_avg'] = True
        
    ### Infer tau_E* from tau_E here rather than in airtable
    # In this case we assume dW/dt = 0 and assume tau_E* = tau_E
    # This case does not trigger a "#" flag on the data table.
    if pd.isnull(row['tau_E_star']) and not pd.isnull(row['tau_E']):
        row['tau_E_star'] = row['tau_E']
        row['inferred_tau_E_star_from_tau_E'] = False
    # In this case we have separately calculated tau_E_star for use.
    # Note that both tau_E and tau_E* must be reported and be different
    # in order to flag the "#" superscript
    if not pd.isnull(row['tau_E_star']) and \
       not pd.isnull(row['tau_E']) and \
       row['tau_E'] != row['tau_E_star']:
        row['inferred_tau_E_star_from_tau_E'] = True
    #print(row['tau_E'], row['tau_E_star'], row['inferred_tau_E_star_from_tau_E'])

    ### Calculate the lawson parameter
    row['ntauEstar_max'] = row['n_i_max'] * row['tau_E_star']
    
    ### Calculate the triple product
    row['nTtauEstar_max'] = float(row['T_i_max']) * row['n_i_max'] * row['tau_E_star']
    return row

mcf_df = mcf_df.apply(process_mcf_experimental_result, axis=1)
#mcf_df

# %% [markdown]
# ### Make LaTeX dataframe for MCF experimental data and create table file

# %% jupyter={"source_hidden": true}
# Handle custom formatting, both asterisks, daggers, significant figures, scientific notation, citations, etc.

# header keys are the dataframe headers
header_keys = {**mcf_airtable_latex_map, **mcf_calculated_latex_map}.keys()
# header values are the corresponding latex table headers
header_values = {**mcf_airtable_latex_map, **mcf_calculated_latex_map}.values()

def mcf_formatting(row):
    # Round values that were multiplied
    if row['inferred_T_i_max_from_T_i_avg'] == True or row['inferred_T_i_max_from_T_e_avg'] == True:
        row['T_i_max'] = round(row['T_i_max'], 2)
    
    if row['inferred_n_i_max_from_n_i_avg'] == True or row['inferred_n_i_max_from_n_e_avg'] == True:
        row['n_i_max'] = '{:0.2e}'.format(row['n_i_max'])
        
    # Format values
    row['T_i_max'] = latexutils.siunitx_num(row['T_i_max'])
    row['T_i_avg'] = latexutils.siunitx_num(row['T_i_avg'])
    row['T_e_max'] = latexutils.siunitx_num(row['T_e_max'])
    row['T_e_avg'] = latexutils.siunitx_num(row['T_e_avg'])

    row['n_i_max'] = latexutils.siunitx_num(row['n_i_max'])
    row['n_i_avg'] = latexutils.siunitx_num(row['n_i_avg'])
    row['n_e_max'] = latexutils.siunitx_num(row['n_e_max'])
    row['n_e_avg'] = latexutils.siunitx_num(row['n_e_avg'])
    
    row['tau_E'] = latexutils.siunitx_num(row['tau_E'])
    row['tau_E_star'] = latexutils.siunitx_num(row['tau_E_star'])
    
    # This is an attempt to standardize the display of the energy confinement times. It doesn't seem to work.
    #row['tau_E_star'] = r'\num[exponent-mode = fixed, fixed-exponent = 6]{' + str(row['tau_E_star']) + r'}'

    
    row['nTtauEstar_max'] = '{:0.1e}'.format(row['nTtauEstar_max'])
    row['nTtauEstar_max'] = latexutils.siunitx_num(row['nTtauEstar_max'])
    
    row['ntauEstar_max'] = '{:0.1e}'.format(row['ntauEstar_max'])
    row['ntauEstar_max'] = latexutils.siunitx_num(row['ntauEstar_max'])
    #print(row)
    row['Bibtex Strings'] = latexutils.cite(row['Bibtex Strings'])    

    # Logic for adding typographical symbols to convey the inferred values is here!
    # Add asterisks to inferred ion temperatures. Note elif is not used as
    # these are all independent conditions (though never can more than one be true per row).
    if row['inferred_T_i_max_from_T_e_max'] == True:
        row['T_i_max'] += r'$^{\dagger}$'
    if row['inferred_T_i_max_from_T_i_avg'] == True:
        row['T_i_max'] += r'$^*$'
    if row['inferred_T_i_max_from_T_e_avg'] == True:
        row['T_i_max'] += r'$^{\dagger *}$'
    if row['inferred_n_i_max_from_n_e_max'] == True:
        row['n_i_max'] += r'$^{\ddagger}$'
    if row['inferred_n_i_max_from_n_i_avg'] == True:
        row['n_i_max'] += r'$^*$'
    if row['inferred_n_i_max_from_n_e_avg'] == True:
        row['n_i_max'] += r'$^{\ddagger *}$'
    if row['inferred_tau_E_star_from_tau_E'] == True:
        row['tau_E_star'] += r'$^{\#}$'
    
    return row

latex_mcf_df = mcf_df.apply(mcf_formatting, axis=1)

mcf_table_footnote = r"""\\$*$ Peak value of density or temperature has been inferred from volume-averaged value as described in Sec.~\ref{sec:inferring_peak_from_average}.\\
$\dagger$ Ion temperature has been inferred from electron temperature as described in Sec.~\ref{sec:inferring_ion_quantities_from_electron_quantities}.\\
$\ddagger$ Ion density has been inferred from electron density as described in Sec.~\ref{sec:inferring_ion_quantities_from_electron_quantities}.\\
$\#$ Energy confinement time $\tau_E^*$ (TFTR/Lawson method) has been inferred from a measurement of the energy confinement time $\tau_E$ (JET/JT-60) method as described in Sec.~\ref{sec:accounting_for_transient_effects}."""

mcf_table_footnote_fixed_references = r"""\\$*$ Peak value of density or temperature has been inferred from volume-averaged value as described in Sec.~IV A 4  of the original paper. \cite{2022_Wurzel_Hsu}\\
$\dagger$ Ion temperature has been inferred from electron temperature as described in Sec.~IV A 5 of the original paper. \cite{2022_Wurzel_Hsu}\\
$\ddagger$ Ion density has been inferred from electron density as described in Sec.~IV A 5 of the original paper. \cite{2022_Wurzel_Hsu}\\
$\#$ Energy confinement time $\tau_E^*$ (TFTR/Lawson method) has been inferred from a measurement of the energy confinement time $\tau_E$ (JET/JT-60) method as described in Sec.~IV A 6 of the original paper. \cite{2022_Wurzel_Hsu}"""

# Only display these headers. ORDER MUST MATCH!
mcf_columns_to_display = [
    'Project Displayname',
    'Concept Displayname',
    'Display Date',
    'Shot',
    'Bibtex Strings',
    'T_i_max',
    #'T_i_avg',
    'T_e_max',
    #'T_e_avg',
    'n_i_max',
    #'n_i_avg',
    'n_e_max',
    #'n_e_avg',
    #'Z_eff',
    'tau_E_star',
    #'tau_E',
    'ntauEstar_max',
    'nTtauEstar_max',
]

## Split into multiple MCF tables since there are too many rows for one page
table_list = [{'concepts': ['Tokamak', 'Spherical Tokamak'],
              'caption': 'Data for tokamaks and spherical tokamaks.',
              'label': 'tab:mainstream_mcf_data_table',
              'filename': 'data_table_mcf_mainstream.tex',
              'filename_csv': 'tables_csv/mcf_mainstream.csv',
              },
              {'concepts': ['Stellarator', 'FRC', 'RFP', 'Z Pinch', 'Pinch', 'Mirror', 'Spheromak', 'MTF'],
              'caption': 'Data for other MCF (i.e. not tokamaks or spherical tokamaks) and lower-density MIF concepts.',
              'label': 'tab:alternates_mcf_data_table',
              'filename': 'data_table_mcf_alternates.tex',
              'filename_csv': 'tables_csv/mcf_alternates.csv',
              },
             ]

for table_dict in table_list:
    concept_latex_mcf_df = latex_mcf_df[latex_mcf_df['Concept Displayname'].isin(table_dict['concepts'])]    
    # Filter the data to only show what is desired
    header_map = {**mcf_airtable_latex_map, **mcf_calculated_latex_map}
    display_header_map = {}
    for header in header_map:
        if header in mcf_columns_to_display:
            display_header_map[header] = header_map[header]
    filtered_concept_latex_mcf_df = concept_latex_mcf_df.filter(items=mcf_columns_to_display)
    
    # Rename the columns of the DataFrame for printing
    filtered_concept_latex_mcf_df = filtered_concept_latex_mcf_df.rename(columns=display_header_map)    
    
    latexutils.latex_table_to_csv(filtered_concept_latex_mcf_df, table_dict['filename_csv'])

    mcf_table_latex = filtered_concept_latex_mcf_df.to_latex(
                      caption=table_dict['caption'],
                      label=table_dict['label'],
                      escape=False,
                      index=False,
                      formatters={},
                      na_rep=latexutils.table_placeholder,
                      )
    mcf_table_latex = latexutils.JFE_comply(mcf_table_latex)
    mcf_table_latex = latexutils.full_width_table(mcf_table_latex)
    mcf_table_latex = latexutils.sideways_table(mcf_table_latex)
    #mcf_table_latex = latexutils.include_table_footnote(mcf_table_latex, mcf_table_footnote)
    mcf_table_latex = latexutils.include_table_footnote(mcf_table_latex, mcf_table_footnote_fixed_references)
    fh=open(os.path.join('tables_latex', label_filename_dict[table_dict['label']]), 'w')
    fh.write(mcf_table_latex)
    fh.close()


# %% [markdown]
# ### Adjust MIF and ICF values so they can be combined with MCF data
#

# %% jupyter={"source_hidden": true}
# Adjust and infer MIF and ICF values so they can be combined with MCF data

def adjust_icf_mif_result(row):
    # The FIREX adjustment is called out in Section IV.B.2 "Inferring Lawson paramter from inferred pressure and confinement dynamics"
    # The other adjustments are necessitated by limited profile data for ICF experiments
    if row['Project Displayname'] == 'FIREX':
        row['T_i_max'] = row['T_e_avg']
    else:
        row['T_i_max'] = row['T_i_avg']
    row['nTtauEstar_max'] = row['nTtauE_avg']
    row['ntauEstar_max'] = row['ntauE_avg']

    return row

icf_mif_df = icf_mif_df.apply(adjust_icf_mif_result, axis=1)
#icf_mif_df
#mcf_df

# %% [markdown]
# ### Merge `mcf_df`, `mif_df` and `icf_df` so they can be plotted together

# %% jupyter={"source_hidden": true}
# Because merging fails with unhashable list object, we drop the Bibtex Strings column before merging
icf_mif_df_no_bibtex = icf_mif_df.drop(columns=['Bibtex Strings'])
mcf_df_no_bibtex = mcf_df.drop(columns=['Bibtex Strings'])

icf_mif_df_no_bibtex['Date'] = pd.to_datetime(icf_mif_df_no_bibtex['Date'])
mcf_df_no_bibtex['Date'] = pd.to_datetime(mcf_df_no_bibtex['Date'])

mcf_mif_icf_df = mcf_df_no_bibtex.merge(icf_mif_df_no_bibtex, how='outer')

# Before plotting we convert all fields which are kept as strings (to maintain sigfigs for tables) to floats for plotting
mcf_mif_icf_df['T_i_max'] = mcf_mif_icf_df['T_i_max'].astype(float)
pd.set_option('display.max_rows', None)    # Show all rows
#mcf_mif_icf_df

# %% [markdown]
# ## Global Plotting Configuration

# %% jupyter={"source_hidden": true}
# #%matplotlib widget
# Use standard matplotlib color cycle for color pallette
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
blue = colors[0]
orange = colors[1]
green = colors[2]
red = colors[3]
purple = colors[4]
brown = colors[5]
pink = colors[6]
grey = colors [7]
lime = colors [8]
teal = colors [9]
black = 'black'

concept_dict = {'Tokamak': {'color': red,
                            'marker': 'o',
                            'markersize': 70,
                           },
                'Stellarator': {'color': green,
                                'marker': '*',
                                'markersize': 200,
                               },
                'RFP': {'color': orange,
                        'marker': 'v',
                        'markersize': 70,
                       },
                'Z Pinch': {'color': blue,
                            'marker': '|',
                            'markersize': 70,
                           },
                'MagLIF': {'color': purple,
                           'marker': '2',
                           'markersize': 70,
                           },
                'FRC': {'color': teal,
                        'marker': 'd',
                        'markersize': 70,
                       },
                'MTF': {'color': purple,
                        'marker': 'o',
                        'markersize': 70,
                       },
                'Spheromak': {'color': pink,
                              'marker': 's',
                              'markersize': 70,
                             },
                'Pinch': {'color': lime,
                          'marker': 'X',
                          'markersize': 70,
                         },
                'Mirror': {'color': brown,
                           'marker': '_',
                           'markersize': 70,
                          },
                'Spherical Tokamak': {'color': grey,
                                      'marker': 'p',
                                      'markersize': 70,
                                     },
                'Laser Indirect Drive': {'color': black,
                                         'marker': 'x',
                                         'markersize': 40,
                                        },
                'Laser Direct Drive': {'color': grey,
                                       'marker': '.',
                                       'markersize': 70,
                                      }
                }

concept_list = concept_dict.keys()
#concept_list = ['Tokamak', 'Laser Indirect Drive', 'Laser Indirect Drive', 'Stellarator', 'MagLIF', 'Spherical Tokamak', 'Z Pinch', 'FRC', 'Spheromak', 'Mirror', 'RFP', 'Pinch'] 

point_size = 70       
alpha = 1
arrow_width = 0.9

ntau_default_indicator = {'arrow': False,
                          'xoff': 0.05,
                          'yoff': -0.07}

# lower MCF band
mcf_ex1 = experiment.LowImpurityPeakedAndBroadDTExperiment()
# upper MCF band
mcf_ex2 = experiment.HighImpurityPeakedAndBroadDTExperiment()

# Plot Q_fuel or Q_sci
#q_type = 'fuel'
q_type = 'sci'


def Q_to_alpha(Q):
    """This function translates a gain Q to a transparency level alpha
    for the purposes of generated plots. The function and constants A and B
    were developed by trial and error to come up with something which looks
    reasonable to the eye.
    """
    A = 0.6
    B = 0.3
    alpha = 1 - (1 / (1 + (A * (Q**B))))
    return alpha

# MCF bands to display
Q_list = {float('inf'), 10, 2, 1, 0.1, 0.01, 0.001}
mcf_bands = []
for Q in Q_list:
    mcf_bands.append({'Q': Q,
                      'color': 'red',
                      'label': r'$Q_{\rm ' + q_type + r'}^{\rm MCF} = ' + str(Q) + r'$',
                      'alpha': Q_to_alpha(Q),
                     })

# Change ICF curve to use betti correction factor
#icf_ex = experiment.IndirectDriveICFDTExperiment()
icf_ex = experiment.IndirectDriveICFDTBettiCorrectionExperiment()

# %% [markdown]
# <HR>

# %% [markdown]
# # Original Plots

# %% [markdown]
# ## Scientific gain vs year achieved

# %% jupyter={"source_hidden": true}
from datetime import date, timedelta
from matplotlib.dates import date2num, num2date
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.dates as mdates

annotation_text_size = 11

class BraceAnnotation:
    def __init__(self, ax, text, x_date, y_pos, width_days, leg_height, head_height, line_width, text_size=annotation_text_size):
        self.ax = ax
        self.text = text
        self.x_date = x_date    
        self.y_pos = y_pos
        self.width = width_days
        self.leg_height = leg_height
        self.head_height = head_height
        self.line_width = line_width
        self.text_size = text_size
        
        # Convert date to matplotlib number for calculations
        x = mdates.date2num(x_date)
        width = mdates.date2num(x_date + timedelta(days=width_days)) - mdates.date2num(x_date - timedelta(days=width_days))

        # Define the bracket vertices
        verts = [
            (x - width/2, y_pos),           # Left end
            (x - width/2, y_pos + leg_height),  # Left top
            (x + width/2, y_pos + leg_height),  # Right top
            (x + width/2, y_pos),           # Right end
            (x, y_pos + leg_height),          # Center start (middle of horizontal line)
            (x, y_pos + leg_height + head_height)    # Center end (to top)
        ]
        
        # Define the path codes
        codes = [
            Path.MOVETO,      # Start at left end
            Path.LINETO,      # Draw to left top
            Path.LINETO,      # Draw to right top
            Path.LINETO,      # Draw to right end
            Path.MOVETO,      # Move to center bottom (without drawing)
            Path.LINETO       # Draw center line
        ]
        
        # Create and add the path
        path = Path(verts, codes)
        patch = PathPatch(path, facecolor='none', edgecolor='black', lw=line_width)
        ax.add_patch(patch)
        
        # Add the text
        ax.text(x, y_pos + leg_height + head_height, text,
                horizontalalignment='center',
                verticalalignment='bottom',
                size=text_size)



with plt.style.context('./styles/large.mplstyle', after_reset=True):
    # Setup figure
    fig, ax = plt.subplots(dpi=dpi)
    fig.set_size_inches(figsize_fullpage)
    
    # Configure axes
    ax.set_ylim(0, 3)
    ax.set_xlim(date(1990, 1, 1), date(2025, 1, 1))
    ax.set_yscale('linear')
    ax.set_xlabel(r'Year')
    ax.set_ylabel(r'$Q_{\rm sci}$')
    ax.grid(which='major')
        
    # Set width to about 2 month in days
    width = timedelta(days=60)
    
    # Plot bars by concept
    for concept in concept_list:
        concept_q_sci_df = q_sci_df[q_sci_df['Concept Displayname'] == concept]
        concept_q_sci_df = concept_q_sci_df[concept_q_sci_df['Q_sci'].notna()]
        if len(concept_q_sci_df) > 0:
            ax.bar(concept_q_sci_df['Date'],
                    concept_q_sci_df['Q_sci'],
                    width=width,
                    color=concept_dict[concept]['color'],
                    label=concept)
    """
    # Annotate all shots directly
    for index, row in q_sci_df.iterrows():
        ax.annotate(
            f"{row['Project Displayname']}",
            xy=(row['Date'], row['Q_sci']),
            xytext=(row['Date'] - timedelta(days=70), row['Q_sci'] + 0.05),
            rotation=90,
            fontsize=annotation_text_size
        )
    """
    # Annotate some shots directly. Don't annotate JET 99971 because it's on top of JET 99972.
    shots_to_annotate_directly = ['26148', '42976', '99972']
    direct_annotate_df = q_sci_df[q_sci_df['Shot'].isin(shots_to_annotate_directly)]
    for index, row in direct_annotate_df.iterrows():
        ax.annotate(
            f"{row['Project Displayname']}",
            xy=(row['Date'], row['Q_sci']),
            xytext=(row['Date'] - timedelta(days=70), row['Q_sci'] + 0.05),
            rotation=90,
            fontsize=annotation_text_size
            )   
    
    # Annotate some shots with arrows (OMEGA)
    shots_to_annotate_with_arrows = ['102154']
    arrow_annotate_df = q_sci_df[q_sci_df['Shot'].isin(shots_to_annotate_with_arrows)]
    for index, row in arrow_annotate_df.iterrows():
        ax.annotate(
            f"{row['Project Displayname']}",
            xy=(row['Date'], row['Q_sci']),
            xytext=(row['Date'] - timedelta(days=4.5*360), row['Q_sci'] + 0.3),
            rotation=0,
            fontsize=annotation_text_size,
            arrowprops={'arrowstyle': '->',
                        'lw': arrow_width,
                       }
            )   

    # Annotate some shots with braces
    BraceAnnotation(ax, 'TFTR', x_date=date(1994, 11, 1), y_pos=0.3, width_days=270, leg_height=0.05, head_height=0.04, line_width=1)
    BraceAnnotation(ax, 'NIF', x_date=date(2022, 6, 1), y_pos=2.4, width_days=700, leg_height=0.05, head_height=0.05, line_width=1)
    BraceAnnotation(ax, 'NIF', x_date=date(2016, 10, 1), y_pos=0.03, width_days=3.5*365, leg_height=0.05, head_height=0.05, line_width=1)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # Add inset for log-linear version
    inset_ax = inset_axes(ax, width="50%", height="50%", bbox_to_anchor=(-0.433, 0.06, 1, 0.9), bbox_transform=ax.transAxes)
    for concept in concept_list:
        concept_q_sci_df = q_sci_df[q_sci_df['Concept Displayname'] == concept]
        concept_q_sci_df = concept_q_sci_df[concept_q_sci_df['Q_sci'].notna()]
        if len(concept_q_sci_df) > 0:
            inset_ax.bar(concept_q_sci_df['Date'],
                    concept_q_sci_df['Q_sci'],
                    width=width,
                    color=concept_dict[concept]['color'],
                    label=concept,
                    zorder=10)
    inset_ax.set_yscale('log')
    # Add horizontal grid lines at major ticks
    inset_ax.yaxis.grid(True, which='major', linewidth=0.8, zorder=0)
    # Set the y-axis formatter to plain numbers (not scientific notation)
    inset_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    inset_ax.tick_params(labelsize=9)



    # Add legend
    ax.legend()
    # Prepublication Watermark
    if add_prepublication_watermark:
        ax.annotate('Prepublication', (date(1995, 1, 1), 0.5), alpha=0.1, size=60, rotation=45)
    plt.tight_layout()
    fig.savefig(os.path.join('images', 'Qsci_vs_year'), bbox_inches='tight')


# %% [markdown]
# ## Lawson parameter vs ion temperature

# %% [markdown]
# ### Function to create a rectanble around a point

# %% jupyter={"source_hidden": true}
def add_rectangle_around_point(ax, x_center, y_center, L_pixels, color='gold', linewidth=2, zorder=10):
    """
    Add a rectangle centered around a point on a plot.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object to draw on
    x_center : float, datetime, or pandas Timestamp
        The x-coordinate of the center point
    y_center : float
        The y-coordinate of the center point
    L_pixels : float
        The size of the rectangle in pixels
    color : str, optional
        The color of the rectangle border
    linewidth : float, optional
        The width of the rectangle border
    zorder : int, optional
        The z-order of the rectangle (higher numbers appear on top)
    """
    # Convert center point to axis coordinates based on x-axis type
    if ax.get_xscale() == 'log':
        x_center_axis = (np.log10(x_center) - np.log10(ax.get_xlim()[0])) / (np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0]))
    else:
        # Handle datetime, Timestamp, or linear x-axis
        x_min, x_max = ax.get_xlim()
        # Convert pandas Timestamp or datetime to matplotlib's numeric format
        if hasattr(x_center, 'timestamp') or isinstance(x_center, datetime):
            # Get the actual datetime limits from the axis
            x_min, x_max = mdates.num2date(ax.get_xlim())  # Convert current limits to datetime
            x_min_num = mdates.date2num(x_min)
            x_max_num = mdates.date2num(x_max)
            x_center_num = mdates.date2num(x_center)
            x_center_axis = (x_center_num - x_min_num) / (x_max_num - x_min_num)
        else:
            # Linear numeric x-axis
            x_center_axis = (x_center - x_min) / (x_max - x_min)
    
    # Handle y-axis scale
    if ax.get_yscale() == 'log':
        y_center_axis = (np.log10(y_center) - np.log10(ax.get_ylim()[0])) / (np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0]))
    else:
        y_center_axis = (y_center - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    # Get the figure size in pixels
    fig_width_pixels = ax.figure.get_dpi() * ax.figure.get_figwidth()
    fig_height_pixels = ax.figure.get_dpi() * ax.figure.get_figheight()

    # Convert pixel length to axis coordinates
    L_axis_x = L_pixels / fig_width_pixels
    L_axis_y = L_pixels / fig_height_pixels

    # Calculate rectangle position and size
    x1_axis = x_center_axis - L_axis_x/2
    y1_axis = y_center_axis - L_axis_y/2
    width = L_axis_x
    height = L_axis_y

    # Create the rectangle
    rectangle = Rectangle((x1_axis, y1_axis),
                         width,
                         height,
                         fill=False,
                         color=color,
                         linewidth=linewidth,
                         transform=ax.transAxes,
                         zorder=zorder)

    ax.add_patch(rectangle)
    return rectangle


# %% [markdown]
# ### Plot of ntau vs T

# %% jupyter={"source_hidden": true}
def plot_ntau_vs_T(on_or_before_date=None,
                   filename=os.path.join('images', label_filename_dict['fig:scatterplot_ntauE_vs_T']),
                   display=True,
                   width=None):
    """
    Plots ntau vs T with optional filters. It's a function so it can be leveraged for animations.

    Parameters:
    - on_or_before_year: Filter data before this year.
    - filename: Filename to save the plot.
    - display: Whether to display the plot.
    - width: Width of the plot.
    """
    ntauE_indicators = {
        'Alcator A': {'arrow': True,
                    'xoff': -0.65,
                    'yoff': 0},
        'Alcator C': {'arrow': True,
                    'xoff': -0.65,
                    'yoff': -0.2},
        'ASDEX': {'arrow': True,
                'xoff': -0.55,
                'yoff': 0},
        'ASDEX-U': {'arrow': True,
                    'xoff': 0.05,
                    'yoff': -0.35},
        'C-2W': {'arrow': True,
                'xabs': 1.8,
                'yabs': 8e15},
        'C-Mod': {'arrow': True,
                'xabs': 3.7,
                'yabs': 3.0e19},
        'DIII-D': {'arrow': True,
                'xabs': 8,
                'yabs': 5e19},
        'EAST': {'arrow': True,
                'xabs': 3.2,
                'yabs': 3e18},
        'ETA-BETA II': {'arrow': True,
                        'xabs': 0.015,
                        'yabs': 3e16},
        'FIREX': {'arrow': True,
                'xabs': 0.6,
                'yabs': 1.1e20},
        'FRX-L': {'arrow': True,
                'xabs': 0.08,
                'yabs': 3e17},
        'FuZE': {'arrow': True,
                'xabs': 3,
                'yabs': 2.2e17},
        'GDT': {'arrow': True,
                'xoff': -0.1,
                'yoff': -0.4},
        'Globus-M2': {'arrow': True,
                    'xabs': 0.2,
                    'yabs': 0.9e18},
        'GOL-3': {'arrow': True,
                'xabs': 3,
                'yabs': 7e17},
        'IPA': {'arrow': True,
                'xabs': 1.5,
                'yabs': 5e16},
        'ITER': {'arrow': True,
                'xabs': 10,
                'yabs': 1e20},
        'JET': {'arrow': True,
                'xabs': 20,
                'yabs': 3e18},
        'JT-60U': {'arrow': True,
                'xabs': 21,
                'yabs': 6e19},
        'KSTAR': {'arrow': True,
                'xabs': 1.3,
                'yabs': 9e18},
        'LHD': {'arrow': True,
                'xabs': 0.2,
                'yabs': 1e20},
        'LSX': {'arrow': True,
                'xabs': 0.23,
                'yabs': 0.9e17},
        'MagLIF': {'arrow': True,
                'xabs': 1.3,
                'yabs': 3.9e19},
        'MAST': {'arrow': True,
                'xoff': 0.15,
                'yoff': 0.},
        'MST': {'arrow': True,
                'xabs': 0.6,
                'yabs': 8e16},
        'NIF': {'arrow': True,
                'xabs': 6,
                'yabs': 3e20},
        'NOVA': {'arrow': True,
                'xabs': 0.3,
                'yabs': 2.5e20},
        'NSTX': {'arrow': True,
                'xoff': -0.25,
                'yoff': 0.60},
        'OMEGA': {'arrow': True,
                'xabs': 1.6,
                'yabs': 7e20},
        'PCS': {'arrow': True,
                'xabs': 0.8,
                'yabs': 1e16},
        'PI3': {'arrow': True,
                'xoff': 0.15,
                'yoff': 0.07},
        'PLT': {'arrow': True,
                'xabs': 1.2,
                'yabs': 5.5e18},
        'RFX-mod': {'arrow': True,
                    'xabs': 4,
                    'yabs': 8e16},
        'SPARC': {'arrow': True,
                'xabs': 25,
                'yabs': 1e20},
        'SSPX': {'arrow': True,
                'xoff': 0.2,
                'yoff': 0.18},
        'ST': {'arrow': True,
            'xabs': 0.25,
            'yabs': 2e17},
        'START': {'arrow': True,
                'xoff': -0.4,
                'yoff': 0.3},
        'T-3': {'arrow': True,
                'xoff': -0.3,
                'yoff': 0.0},
        'TFR': {'arrow': True,
                'xabs': 0.4,
                'yabs': 5e18},
        'TFTR': {'arrow': True,
                'xabs': 40,
                'yabs': 3e18},
        'W7-A': {'arrow': True,
                'xoff': -0.5,
                'yoff': -0.02},
        'W7-X': {'arrow': True,
                'xabs': 1.4,
                'yabs': 1.5e19},
        'Yingguang-I': {'arrow': True,
                        'xoff': -0.14,
                        'yoff': -0.55},
        'ZETA': {'arrow': True,
                'xabs': 0.03,
                'yabs': 1e16},
        'ZT-40M': {'arrow': True,
                'xabs': 0.25,
                'yabs': 3.1e16}
    }

    # Ignition ICF curve
    icf_curves = [{'Q':float('inf'),
                'dashes':  (1, 0),
                'linewidth': '0.1',
                'color': 'black',
                'alpha' : 1,
                #'label': r'placeholder',
                }]

    # This is needed for the correct ordering of the legend entries
    legend_handles = []

    with plt.style.context(['./styles/large.mplstyle'], after_reset=True):
        fig, ax = plt.subplots(dpi=dpi)
        fig.set_size_inches(figsize_fullpage)

        xmin = 0.01 # keV
        xmax = 100  # keV
        ax.set_xlim(xmin, xmax)
        ymin = 1e14
        ymax = 1e22
        ax.set_ylim(ymin, ymax)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel(r'$T_{i0}, \langle T_i \rangle_{\rm n} \; {\rm (keV)}$')
        ax.set_ylabel(r'$n_{i0} \tau_E^*, \; n \tau_{\rm stag} \; {\rm (m^{-3}~s)}$')
        ax.grid('on', which='major', axis='both')
        #ax.set_title('Lawson Parameter vs Ion Temperature', size=16)

        ##### MCF Bands
        for mcf_band in mcf_bands:
            if mcf_band['Q'] < 1:
                edgecolor = mcf_band['color']
            else:
                edgecolor = 'none'
            handle = ax.fill_between(DT_requirements_df['T_i0'],
                            DT_requirements_df[mcf_ex1.name + '__ntauE_Q_' + q_type + '=' + str(mcf_band['Q'])],
                            DT_requirements_df[mcf_ex2.name + '__ntauE_Q_' + q_type + '=' + str(mcf_band['Q'])],
                            color=mcf_band['color'],
                            #label=mcf_band['label'],
                            label='_hidden' + mcf_band['label'],
                            zorder=0,
                            alpha=mcf_band['alpha'],
                            edgecolor=edgecolor,
                        )
            legend_handles.append(handle)
        
        ##### ICF curve
        for icf_curve in icf_curves:
            handle = ax.plot(DT_requirements_df['T_i0'],
                            DT_requirements_df[icf_ex.name + '__ntauE_Q_' + q_type + '=' + str(icf_curve['Q'])],                                           linewidth=1,
                            color=icf_curve['color'],
                            alpha=icf_curve['alpha'],
                            dashes=icf_curve['dashes'],
                            )
            legend_handles.append(handle[0])

        ##### Scatterplot
        for concept in concept_list:
            if on_or_before_date is None:
                concept_df = mcf_mif_icf_df[mcf_mif_icf_df['Concept Displayname']==concept]
            else:
                concept_df = mcf_mif_icf_df[(mcf_mif_icf_df['Concept Displayname']==concept) & (mcf_mif_icf_df['Date']<=on_or_before_date)] 
            if concept_dict[concept]['marker'] not in ['|', '2', '_', 'x']:
                edgecolor='white'
            else:
                edgecolor=None
            handle = ax.scatter(concept_df['T_i_max'],
                                concept_df['ntauEstar_max'], 
                                c = concept_dict[concept]['color'], 
                                marker = concept_dict[concept]['marker'],
                                s = concept_dict[concept]['markersize'],
                                edgecolors= edgecolor,
                                zorder=10,
                                label=concept,
                            )
            #legend_handles.append(handle)
            # Annotate data points
            for index, row in concept_df.iterrows():
                displayname = row['Project Displayname']
                ntauE_indicator = ntauE_indicators.get(displayname, ntau_default_indicator)
                text = row['Project Displayname']
                if text in ['SPARC', 'ITER']:
                    text += '*'
                annotation = {'text': text,
                            'xy': (row['T_i_max'], row['ntauEstar_max']),
                            }
                if ntauE_indicator['arrow'] is True:
                    annotation['arrowprops'] = {'arrowstyle': '->',
                                                'lw': arrow_width,
                                            }
                else:
                    pass
                if 'xabs' in ntauE_indicator:
                    # Annotate with absolute placement
                    annotation['xytext'] = (ntauE_indicator['xabs'], ntauE_indicator['yabs'])
                else:
                    # Annotate with relative placement accounting for logarithmic scale
                    annotation['xytext'] = (10**ntauE_indicator['xoff'] * row['T_i_max'], 10**ntauE_indicator['yoff'] * row['ntauEstar_max'])
                annotation['zorder'] = 10
                ax.annotate(**annotation)
        
        # Draw rectangle around N210808 to highlight that it achieved ignition and is termimal data point
        if on_or_before_date is None or on_or_before_date > datetime(2021, 8, 8):
            n210808_data = mcf_mif_icf_df[mcf_mif_icf_df['Shot'] == 'N210808']
            x_center, y_center = n210808_data['T_i_max'].iloc[0], n210808_data['ntauEstar_max'].iloc[0]
            add_rectangle_around_point(ax, x_center, y_center, L_pixels=50)

        # Custom format temperature axis
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(latexutils.CustomLogarithmicFormatter))
        
        ### ANNOTATIONS
        # Prepublication Watermark
        if add_prepublication_watermark:
            ax.annotate('Prepublication', (0.02, 1.5e15), alpha=0.1, size=60, rotation=45)
        
        # Right side annotations
        annotation_offset = 5
        ax.annotate(r'$Q_{\rm sci}^{\rm MCF}$', xy=(xmax, ymax), xytext=(xmax+annotation_offset, 6e20), xycoords='data', alpha=1, color='red', rotation=0)
        horiz_line = mpl.patches.Rectangle((1.005, 0.83),
                                    width=0.06,
                                    height=0.002,
                                    transform=ax.transAxes,
                                    color='red',
                                    clip_on=False
                                    )
        ax.add_patch(horiz_line)
        ax.annotate(r'$\infty$', xy=(xmax, ymax), xytext=(xmax+annotation_offset, 3.1e20), xycoords='data', alpha=1, color='red', rotation=0)
        ax.annotate(r'$10$', xy=(xmax, ymax), xytext=(xmax+annotation_offset, 1.8e20), xycoords='data', alpha=1, color='red', rotation=0)
        ax.annotate(r'$2$', xy=(xmax, ymax), xytext=(xmax+annotation_offset, 8.5e19), xycoords='data', alpha=1, color='red', rotation=0)
        ax.annotate(r'$1$', xy=(xmax, ymax), xytext=(xmax+annotation_offset, 4.6e19), xycoords='data', alpha=1, color='red', rotation=0)
        ax.annotate(r'$0.1$', xy=(xmax, ymax), xytext=(xmax+annotation_offset, 5e18), xycoords='data', alpha=1, color='red', rotation=0)
        ax.annotate(r'$0.01$', xy=(xmax, ymax), xytext=(xmax+annotation_offset, 5.5e17), xycoords='data', alpha=1, color='red', rotation=0)
        ax.annotate(r'$0.001$', xy=(xmax, ymax), xytext=(xmax+annotation_offset, 5.5e16), xycoords='data', alpha=1, color='red', rotation=0)
        
        # Inner annotations
        ax.annotate(r'$(n \tau_{\rm stag})_{\rm ig, hs}^{\rm ICF}$', xy=(xmax, ymax), xytext=(25, 4.9e20), xycoords='data', alpha=1, color='black', rotation=25)
        
        # Only show "* Maximum projected" if the year is greater than the current year or if no year is being displayed.
        if on_or_before_date is None or on_or_before_date.year > datetime.now().year:
            ax.annotate('* Maximum projected', xy=(xmax, ymax), xytext=(10.2, 1.2e14), xycoords='data', alpha=1, color='black', size=10)

        # Show the year on the bottom right if a specific year is requested
        if on_or_before_date is not None:
            ax.annotate(f'{on_or_before_date.year}', (12, 1.7e15), alpha=0.8, size=40)
            if on_or_before_date.year > 2025:
                ax.annotate('(projected)', (10, 4e14), alpha=0.8, size=22)
                ax.annotate('* Maximum projected', xy=(xmax, ymax), xytext=(10.2, 1.2e14), xycoords='data', alpha=1, color='black', size=10)

        # Legend to the right
        #plt.legend(legend_handles,[H.get_label() for H in legend_handles],
        #            bbox_to_anchor=(1, 1.014), ncol=1)
        
        # Legend below
        #plt.legend(legend_handles,[H.get_label() for H in legend_handles],
        #    bbox_to_anchor=(1.04, -0.12), ncol=4)
        #plt.legend(bbox_to_anchor=(1.04, -0.12), ncol=4)
        
        # Legend inside
        leg = ax.legend()
        
        #leg.set_draggable(state=True)
        #fig.canvas.resizable = True
        #plt.show()
        fig.savefig(filename, bbox_inches='tight')
        if not display:
            plt.close(fig)    
plot_ntau_vs_T()
#fig = plot_ntau_vs_T(on_or_before_date=datetime(2022, 1, 1))

# %% [markdown]
# ## Triple Product vs ion temperature

# %% jupyter={"source_hidden": true}
default_indicator = {'arrow': False,
                     'xoff': 0.05,
                     'yoff': -0.07}

nTtauE_indicators = {
    'Alcator A': {'arrow': True,
                  'xoff': -0.62,
                  'yoff': 0},
    'Alcator C': {'arrow': True,
                  'xabs': 0.3,
                  'yabs': 1e20},
    'ASDEX': {'arrow': True,
              'xoff': -0.60,
              'yoff': -.1},
    'ASDEX-U': {'arrow': True,
                'xoff': 0.2,
                'yoff': -0.25},
    'C-2W': {'arrow': True,
             'xabs': 3,
             'yabs': 1e16},
    'C-Mod': {'arrow': True,
              'xabs': 2.5,
              'yabs': 2.5e19},
    'DIII-D': {'arrow': True,
               'xabs': 6.5,
               'yabs': 3e20},
    'EAST': {'arrow': True,
             'xabs': 3,
             'yabs': 2e18},
    'ETA-BETA I': {'arrow': True,
                   'xoff': 0.08,
                   'yoff': 0.3},
    'ETA-BETA II': {'arrow': True,
                    'xoff': -0.8,
                    'yoff': -.1},
    'FIREX': {'arrow': True,
              'xabs': 1.1,
              'yabs': 5e19},
    'FRX-L': {'arrow': True,
              'xabs': 0.04,
              'yabs': 3.5e16},
    'FuZE': {'arrow': True,
             'xabs': 3,
             'yabs': 4e17},
    'GDT': {'arrow': True,
            'xabs': 0.8,
            'yabs': 1.7e15},
    'Globus-M2': {'arrow': True,
                  'xabs': 0.7,
                  'yabs': 3.5e17},
    'GOL-3': {'arrow': True,
              'xabs': 3,
              'yabs': 1e18},
    'IPA': {'arrow': True,
            'xabs': 2.5,
            'yabs': 3e16},
    'ITER': {'arrow': True,
             'xabs': 15,
             'yabs': 5e22},
    'JET': {'arrow': True,
            'xabs': 20,
            'yabs': 0.5e20},
    'JT-60U': {'arrow': True,
               'xabs': 12,
               'yabs': 1.55e21},
    'KSTAR': {'arrow': True,
              'xabs': 1,
              'yabs': 2.3e19},
    'LSX': {'arrow': True,
            'xabs': 0.7,
            'yabs': 1.2e17},
    'MagLIF': {'arrow': True,
               'xabs': 0.5,
               'yabs': 1e21},
    'MAST': {'arrow': True,
             'xoff': 0.15,
             'yoff': 0.06},
    'MST': {'arrow': True,
            'xabs': 0.2,
            'yabs': 8e17},
    'NIF': {'arrow': True,
            'xabs': 6.5,
            'yabs': 8e20},
    'NOVA': {'arrow': True,
             'xabs': 0.3,
             'yabs': 2e20},
    'NSTX': {'arrow': True,
             'xoff': -0.6,
             'yoff': 0.4},
    'OMEGA': {'arrow': True,
              'xabs': 1.3,
              'yabs': 3e21},
    'PCS': {'arrow': True,
            'xabs': 1.2,
            'yabs': 6e15},
    'PI3': {'arrow': True,
             'xoff': -0.7,
             'yoff': 0.47},    
    'PLT': {'arrow': True,
            'xabs': 1,
            'yabs': 9e18},
    'SPARC': {'arrow': True,
              'xabs': 30,
              'yabs': 2e21},
    'SSPX': {'arrow': True,
             'xoff': -0.8,
             'yoff': 0.39},
    'ST': {'arrow': True,
           'xoff': -0.4,
           'yoff': 0.25},
    'START': {'arrow': True,
              'xoff': -0.4,
              'yoff': 0.36},
    'T-3': {'arrow': True,
            'xoff': -0.4,
            'yoff': -0.3},
    'TCSU': {'arrow': True,
             'xoff': 0.08,
             'yoff': -0.5},
    'TFR': {'arrow': True,
            'xabs': 0.3,
            'yabs': 3e18},
    'TFTR': {'arrow': True,
             'xabs': 50,
             'yabs': 9e19},
    'W7-A': {'arrow': True,
             'xoff': -0.45,
             'yoff': 0.3},
    'W7-AS': {'arrow': True,
              'xoff': 0.15,
              'yoff': -0.06},
    'W7-X': {'arrow': True,
             'xoff': -0.05,
             'yoff': 0.4},
    'ZT-40M': {'arrow': True,
               'xabs': 0.05,
               'yabs': 7e16}
}
# This is needed for the correct ordering of the legend entries
legend_handles = []

# Needed here for custom ICF curve
icf_curves = [{'Q':float('inf'),
               'dashes':  (1, 0),
               'linewidth': '0.1',
               'color': 'black',
               'alpha' : 1,
               'label': r'$(n T \tau)_{\rm ig, hs}^{\rm ICF}$',
              }]

with plt.style.context('./styles/large.mplstyle', after_reset=True):
    fig, ax = plt.subplots(dpi=dpi)
    fig.set_size_inches(figsize_fullpage)

    xmin = 0.01
    xmax = 100
    ax.set_xlim(xmin, xmax)
    ymin = 1e12
    ymax = 1e23
    ax.set_ylim(ymin, ymax)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid('on', which='major', axis='both')
    #ax.set_title('Triple Product vs Ion Temperature', size=16)
    
    ##### MCF Bands
    # In order for ax.fill_between to correctly fill the region that goes to
    # infinity, the values of infinity in the dataframe must be replaced with
    # non-infinite values. We replace the infinities with the values of the
    # maximum y that is plotted here.
    DT_requirements_df = DT_requirements_df.replace(math.inf, ymax)

    for mcf_band in mcf_bands:
        handle = ax.fill_between(DT_requirements_df['T_i0'],
                        DT_requirements_df[mcf_ex1.name + '__nTtauE_Q_fuel=%s' % mcf_band['Q']],
                        DT_requirements_df[mcf_ex2.name + '__nTtauE_Q_fuel=%s' % mcf_band['Q']],
                        color=mcf_band['color'],
                        #label=mcf_band['label'],
                        label='_hiden_' + mcf_band['label'],
                        zorder=0,
                        alpha=mcf_band['alpha'],
                       )
        legend_handles.append(handle)

    ##### ICF Curves   
    for icf_curve in icf_curves:
        handle = ax.plot(DT_requirements_df['T_i0'],
                         DT_requirements_df[icf_ex.name + '__nTtauE_Q_' + q_type + '=' + str(icf_curve['Q'])],                                           linewidth=1,
                         color=icf_curve['color'],
                         #label=icf_curve['label'],
                         label='_hiden_' + mcf_band['label'],
                         alpha=icf_curve['alpha'],
                         dashes=icf_curve['dashes'],
                        )
        legend_handles.append(handle[0])

    ##### Scatterplot

    #for concept in mcf_mif_icf_df['Concept Displayname'].unique():
    for concept in concept_list:
        # Plot points for each concept
        concept_df = mcf_mif_icf_df[mcf_mif_icf_df['Concept Displayname']==concept]

        #project = project_df['Concept Displayname'].iloc[0]
        handle = ax.scatter(concept_df['T_i_max'], concept_df['nTtauEstar_max'], 
                   c = concept_dict[concept]['color'], 
                   marker = concept_dict[concept]['marker'],
                   zorder=10,
                   label=concept,
                   s = concept_dict[concept]['markersize'],
                   edgecolors= 'white',
                  )
        legend_handles.append(handle)
        # Annotate
        for index, row in concept_df.iterrows():
            displayname = row['Project Displayname']
            nTtauE_indicator = nTtauE_indicators.get(displayname, default_indicator)
            text = row['Project Displayname']
            if text in ['SPARC', 'ITER']:
                text += '*'
            annotation = {'text': text,
                          'xy': (row['T_i_max'], row['nTtauEstar_max']),

                         }
            if nTtauE_indicator['arrow'] is True:
                annotation['arrowprops'] = {'arrowstyle': '->'}
            else:
                pass
            if 'xabs' in nTtauE_indicator:
                # Annotate with absolute placement
                annotation['xytext'] = (nTtauE_indicator['xabs'], nTtauE_indicator['yabs'])
            else:
                # Annotate with relative placement
                annotation['xytext'] = (10**nTtauE_indicator['xoff'] * row['T_i_max'], 10**nTtauE_indicator['yoff'] * row['nTtauEstar_max'])
            annotation['zorder'] = 10
            ax.annotate(**annotation)
    
    # Draw rectangle around N210808 to highlight that it achieved ignition and is termimal data point
    n210808_data = mcf_mif_icf_df[mcf_mif_icf_df['Shot'] == 'N210808']
    x_center, y_center = n210808_data['T_i_max'].iloc[0], n210808_data['nTtauEstar_max'].iloc[0]
    add_rectangle_around_point(ax, x_center, y_center, L_pixels=50)
    
    # Format temperature axis
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(latexutils.CustomLogarithmicFormatter))
    
    ### ANNOTATIONS
    # Prepublication Watermark
    if add_prepublication_watermark:
        ax.annotate('Prepublication', (0.02, 1.5e15), alpha=0.1, size=60, rotation=45)
    
    # Right side annotations
    annotation_offset = 5
    ax.annotate(r'$Q_{\rm sci}^{\rm MCF}$', xy=(xmax, ymax), xytext=(xmax+annotation_offset, 8e22), xycoords='data', alpha=1, color='red', rotation=0)
    horiz_line = mpl.patches.Rectangle((1.005, 0.975),
                                 width=0.06,
                                 height=0.002,
                                 transform=ax.transAxes,
                                 color='red',
                                 clip_on=False
                                )
    ax.add_patch(horiz_line)
    ax.annotate(r'$\infty$', xy=(xmax, ymax), xytext=(xmax+annotation_offset, 3.2e22), xycoords='data', alpha=1, color='red', rotation=0)
    ax.annotate(r'$10$', xy=(xmax, ymax), xytext=(xmax+annotation_offset, 1.8e22), xycoords='data', alpha=1, color='red', rotation=0)
    ax.annotate(r'$2$', xy=(xmax, ymax), xytext=(xmax+annotation_offset, 8.5e21), xycoords='data', alpha=1, color='red', rotation=0)
    ax.annotate(r'$1$', xy=(xmax, ymax), xytext=(xmax+annotation_offset, 4e21), xycoords='data', alpha=1, color='red', rotation=0)
    ax.annotate(r'$0.1$', xy=(xmax, ymax), xytext=(xmax+annotation_offset, 6e20), xycoords='data', alpha=1, color='red', rotation=0)
    ax.annotate(r'$0.01$', xy=(xmax, ymax), xytext=(xmax+annotation_offset, 5e19), xycoords='data', alpha=1, color='red', rotation=0)
    ax.annotate(r'$0.001$', xy=(xmax, ymax), xytext=(xmax+annotation_offset, 6e18), xycoords='data', alpha=1, color='red', rotation=0)
    #ax.annotate(r'$10^{-4}$', xy=(xmax, ymax), xytext=(xmax+annotation_offset, 6e17), xycoords='data', alpha=1, color='red', rotation=0)
    
    # Inner annotations
    ax.annotate(r'$(n T \tau_{\rm stag})_{\rm ig, hs}^{\rm ICF}$', xy=(xmax, ymax), xytext=(0.6, 4e22), xycoords='data', alpha=1, color='black', rotation=0)
    ax.annotate('* Maximum projected', xy=(xmax, ymax), xytext=(10.2, 1.3e12), xycoords='data', alpha=1, color='black', size=10)

    
    # Legend to the right
    # plt.legend(legend_handles,[H.get_label() for H in legend_handles],
    #            bbox_to_anchor=(1, 1.014), ncol=1)
        
    # Legend below
    #plt.legend(legend_handles,[H.get_label() for H in legend_handles],
    #           bbox_to_anchor=(1.01, -0.12), ncol=4)
    # Legend inside
    plt.legend()
    
    ax.set_xlabel(r'$T_{i0}, \langle T_i \rangle_{\rm n} \; {\rm (keV)}$')
    ax.set_ylabel(r'$n_{i0} T_{i0} \tau_E^*, \; n \langle T_i \rangle_{\rm n} \tau_{\rm stag} \; {\rm (m^{-3}~keV~s)}$')
    fig.savefig(os.path.join('images', label_filename_dict['fig:scatterplot_nTtauE_vs_T']), bbox_inches='tight')

# %% [markdown]
# ## Triple product vs year achieved

# %% jupyter={"source_hidden": true}
from datetime import datetime 
# Identify triple product results which are records for that particular concept
def is_concept_record(row):
    # Don't directly show projected results
    if row['Date'].year > datetime.now().year:
        return False
    
    concept_displayname = row['Concept Displayname']
    date = row['Date']
    nTtauEstar_max = row['nTtauEstar_max']
    matches = mcf_mif_icf_df.query("`Concept Displayname` == @concept_displayname & \
                                    `Date` <= @date & \
                                    `nTtauEstar_max` > @nTtauEstar_max"
                                 )
    if len(matches.index) == 0:
        return True
    else:
        return False
    
mcf_mif_icf_df['is_concept_record'] = mcf_mif_icf_df.apply(is_concept_record, axis=1)
mcf_mif_icf_df.sort_values(by='Date', inplace=True)

# %% [markdown]
# ### Plot of Triple Product vs Year

# %% jupyter={"source_hidden": true}
default_indicator = {'arrow': False,
                     'xoff': 1,
                     'yoff': 0}
indicators = {
    'Alcator A': {'arrow': False,
                  'xoff': 0,
                  'yoff': -0.3},
    'Alcator C': {'arrow': False,
                  'xoff': -12,
                  'yoff': 0},
    'C-2U': {'arrow': True,
             'xabs': datetime(2014, 1, 1),
             'yabs': 4e17},
    'C-2W': {'arrow': True,
             'xabs': datetime(2025, 1, 1),
             'yabs': 1.6e17},
    'C-Stellarator': {'arrow': True,
                      'xabs': datetime(1963, 1, 1),
                      'yabs': 0.3e14},
    'CTX': {'arrow': False,
            'xoff': 1,
            'yoff': -0.1},
    'ETA-BETA II': {'arrow': True,
                    'xabs': datetime(1957, 1, 1),
                    'yabs': 2.5e15},
    'FuZE': {'arrow': True,
             'xabs': datetime(2027, 1, 1),
             'yabs': 9e17},
    'JET': {'arrow': False,
            'xoff': -6,
            'yoff': 0},
    'JT-60U': {'arrow': False,
               'xoff': 0,
               'yoff': -0.25},
    'LHD': {'arrow': False,
            'xoff': 0,
            'yoff': -0.33},
    'LSX': {'arrow': False,
            'xoff': -1,
            'yoff': 0.2},
    'MagLIF': {'arrow': True,
               'xabs': datetime(2021, 6, 1),
               'yabs': 2e20},
    'MAST': {'arrow': False,
             'xoff': -5,
             'yoff': 0.1},
    'MST': {'arrow': True,
            'xabs': datetime(2010, 1, 1),
            'yabs': 6e15},
    'NIF': {'arrow': True,
            'xabs': datetime(2008, 1, 1),
            'yabs': 5e22},
    'NOVA': {'arrow': False,
             'xoff': 1,
             'yoff': -0.2},
    'NSTX': {'arrow': False,
             'xoff': 0,
             'yoff': 0.2},
    'OMEGA': {'arrow': True,
              'xabs': datetime(2012, 6, 1),
              'yabs': 4e21},
    'PCS': {'arrow': True,
              'xabs': datetime(2025, 1, 1),
              'yabs': 3e16},
    'RFX-mod': {'arrow': True,
                'xabs': datetime(2016, 1, 1),
                'yabs': 1e16},
    'SSPX': {'arrow': True,
             'xabs': datetime(2005, 1, 1),
             'yabs': 4e17},
    'START': {'arrow': True,
              'xabs': datetime(1986, 1, 1),
              'yabs': 3e16},
    'TFTR': {'arrow': False,
             'xoff': -2,
             'yoff': 0.2},
    'TMX-U': {'arrow': False,
              'xabs': datetime(1985, 1, 1),
              'yabs': 2e14},
    'W7-A': {'arrow': True,
             'xoff': 1,
             'yoff': -0.5},
    'W7-AS': {'arrow': False,
              'xoff': -9,
              'yoff': 0.1},
    'ZaP': {'arrow': False,
            'xoff': 1,
            'yoff': -0.1},
    'ZT-40M': {'arrow': False,
               'xoff': -10,
               'yoff': -.1}
}

# mcf_horizontal_range_dict sets the horizontal location and width of the Q_sci^MCF lines.
# The keys are the values of Q_sci^MCF, the list in the values are [start year, length of line in years]
mcf_horizontal_range_dict = {1: [datetime(1950, 1, 1), timedelta(days=365*100)],
                             2: [datetime(1961, 1, 1), timedelta(days=365*100)],
                             10: [datetime(1972, 1, 1), timedelta(days=365*100)],
                             float('inf'): [datetime(1985, 1, 1), timedelta(days=365*100)],
                            }

with plt.style.context('./styles/large.mplstyle', after_reset=True):

    # Generate Figure    
    fig, ax = plt.subplots(dpi=dpi)
    fig.set_size_inches(figsize_fullpage)

    # Set Range
    ymin = 1e12
    ymax = 1e23
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(datetime(1950, 1, 1), datetime(2045, 1, 1))
    ax.set_yscale('log')

    # Label Title and Axes
    #ax.set_title('Record Triple Product by Concept vs Year', size=16)
    ax.set_xlabel(r'Year')
    ax.set_ylabel(r'$n_{i0} T_{i0} \tau_E^*, \; n \langle T_i \rangle_{\rm n} \tau_{\rm stag} \; {\rm (m^{-3}~keV~s)}$')
    ax.grid(which='major')

    # Plot horizontal lines for indicated values of Q_MCF (actually rectangles of height
    # equal to difference between maximum and minimum values of triple product at temperature
    # for which the minimum triple product (proportional to pressure) is required.
    #mcf_qs = [float('inf'), 1]
    mcf_qs = [float('inf'), 10, 2, 1]
    for mcf_band in [mcf_band for mcf_band in mcf_bands if mcf_band['Q'] in mcf_qs]:
        min_mcf_low_impurities = DT_requirement_minimum_values_df.loc[DT_requirement_minimum_values_df['requirement'] == 'lipabdt_experiment__nTtauE_Q_{q_type}={Q}'.format(Q=mcf_band['Q'], q_type=q_type)].iloc[0]['minimum_value'] 
        T_i0_min_mcf_low_impurities = DT_requirement_minimum_values_df.loc[DT_requirement_minimum_values_df['requirement'] == 'lipabdt_experiment__nTtauE_Q_{q_type}={Q}'.format(Q=mcf_band['Q'], q_type=q_type)].iloc[0]['T_i0']

        min_mcf_high_impurities = DT_requirement_minimum_values_df.loc[DT_requirement_minimum_values_df['requirement'] == 'hipabdt_experiment__nTtauE_Q_{q_type}={Q}'.format(Q=mcf_band['Q'], q_type=q_type)].iloc[0]['minimum_value'] 
        T_i0_min_mcf_high_impurities = DT_requirement_minimum_values_df.loc[DT_requirement_minimum_values_df['requirement'] == 'hipabdt_experiment__nTtauE_Q_{q_type}={Q}'.format(Q=mcf_band['Q'], q_type=q_type)].iloc[0]['T_i0']      
        #min_mcf_high_impurities = DT_min_triple_product_df.loc[DT_min_triple_product_df['Q'] == 'peaked_and_broad_high_impurities Q={Q}'.format(Q=Q)].iloc[0]['minimum_triple_product'] 
        
        mcf_patch_height = min_mcf_high_impurities - min_mcf_low_impurities
        mcf_patch = patches.Rectangle(xy=(mcf_horizontal_range_dict.get(mcf_band['Q'], [datetime(1950,1,1)])[0],
                                          min_mcf_low_impurities),
                                          width = mcf_horizontal_range_dict.get(mcf_band['Q'], [0, timedelta(days=365*100)])[1], # width of line in years
                                          height = mcf_patch_height,
                                          linewidth=0,
                                          facecolor=mcf_band['color'],
                                          alpha=mcf_band['alpha'],
                                     )
        ax.add_patch(mcf_patch)
        # print the gain and temperature at which the minimum triple product is achieved for the low and high impurity cases
        print(f"Q={mcf_band['Q']}, T_i0_min_mcf_low_impurities={T_i0_min_mcf_low_impurities:.2f}, T_i0_min_mcf_high_impurities={T_i0_min_mcf_high_impurities:.2f}")
        # annotate the gain and temperature at which the minimum triple product is achieved for the low and high impurity cases
        # Uncomment the below phantom line to display Q_MCF lines in legend
        #legend_string = r'$Q_{{\rm ' + q_type + r'}}^{{\rm MCF}}={' + str(mcf_band['Q']).replace('inf', '\infty') + r'}$'
        #ax.hlines(0, 0, 0, color=mcf_band['color'], alpha=mcf_band['alpha'], 
        #          linestyles="solid", linewidths=3, label=legend_string, zorder=0)
    
    # Draw golden rectangle around N210808 to highlight that it achieved threshold of ignition and is termimal data point for this graph
    n210808_data = mcf_mif_icf_df[mcf_mif_icf_df['Shot'] == 'N210808']
    x_center, y_center = n210808_data['Date'].iloc[0], n210808_data['nTtauEstar_max'].iloc[0]
    add_rectangle_around_point(ax, x_center, y_center, L_pixels=50)

    ax.annotate(r'$T_{i0} \approx 20 \text{ to } 27~\mathrm{keV}$', xy=(datetime(1950, 6, 1), 5e20), color='red')
    
    # Plot horizontal lines and annotations for ICF ignition only assuming T_i=4 keV and T_i=10 keV
    icf_ignition_10keV = icf_ex.triple_product_Q_sci(
                                 T_i0=10.0,
                                 Q_sci=float('inf'),
                                )
    icf_ignition_4keV = icf_ex.triple_product_Q_sci(
                                 T_i0=4.0,
                                 Q_sci=float('inf'),
                                )
    
    ax.hlines(icf_ignition_4keV,
              xmin=datetime(2000,1,1),
              xmax=datetime(2050,1,1),
              color=icf_curve['color'],
              linewidth=2,
              linestyle=(0, icf_curve['dashes']),
              label='_hidden',
              #label=r'$(n T \tau)_{\rm ig}^{\rm ICF}$',
              zorder=9
             )

    ax.hlines(icf_ignition_10keV,
              xmin=datetime(2000,1,1),
              xmax=datetime(2050,1,1),
              #color=icf_curve['color'],
              color='gold',
              linewidth=2,
              linestyle=(0, icf_curve['dashes']),
              label='_hidden',
              #label=r'$(n T \tau)_{\rm ig}^{\rm ICF}$',
              zorder=2
             )
    ax.hlines(icf_ignition_10keV,
              xmin=datetime(2000,1,1),
              xmax=datetime(2050,1,1),
              color='black',
              linewidth=2,
              linestyle=':',
              label='_hidden',
              #label=r'$(n T \tau)_{\rm ig}^{\rm ICF}$',
              zorder=3
             )
    ax.annotate(r'$(n T \tau_{\rm stag})_{\rm ig, hs}^{\rm ICF}$', (datetime(2017,1,1), 4e22), alpha=1, color='black')
    ax.annotate(r'${\rm @ 10~keV}$', (datetime(2033,1,1), 1.25e22), alpha=1, color='black')
    ax.annotate(r'${\rm @ 4~keV}$', (datetime(2033,1,1), 4e22), alpha=1, color='black')
    #ax.annotate(r'$@T_i = 10{\rm keV}$', (datetime(1990,1,1), 3.7e21), alpha=1, color='black')
    # Scatterplot of data
    #d = mcf_mif_icf_df[mcf_mif_icf_df['is_concept_record'] == True]
    # Make exception for N210808 since it achieved hot-spot ignition
    d = mcf_mif_icf_df[
    (mcf_mif_icf_df['is_concept_record'] == True) | 
    (mcf_mif_icf_df['Shot'].isin(['N210808']))
    ]
    #for concept in d['Concept Displayname'].unique():
    for concept in concept_list:
        # Draw datapoints
        concept_df = d[d['Concept Displayname']==concept]
        scatter = ax.scatter(concept_df['Date'], concept_df['nTtauEstar_max'], 
                             c = concept_dict[concept]['color'], 
                             marker = concept_dict[concept]['marker'],
                             zorder=10,
                             s=point_size,
                             label=concept,
                            )
        # Draw lines between datapoints
        plot = ax.plot(concept_df['Date'], concept_df['nTtauEstar_max'], 
                             c = concept_dict[concept]['color'], 
                             marker = concept_dict[concept]['marker'],
                             zorder=10,
                            )
        # Annotate
        for index, row in concept_df.iterrows():
            displayname = row['Project Displayname']
            indicator = indicators.get(displayname, default_indicator)
            annotation = {'text': row['Project Displayname'],
                          'xy': (row['Date'], row['nTtauEstar_max']),

                         }
            if indicator['arrow'] is True:
                annotation['arrowprops'] = {'arrowstyle': '->'}
            else:
                pass
            if 'xabs' in indicator:
                # Annotate with absolute placement
                annotation['xytext'] = (indicator['xabs'], indicator['yabs'])
            else:
                # Annotate with relative placement
                annotation['xytext'] = (row['Date'] + timedelta(days=365*indicator['xoff']), 10**indicator['yoff'] * row['nTtauEstar_max'])
            annotation['zorder'] = 10
            ax.annotate(**annotation)

    #SPARC
    sparc_tp = mcf_mif_icf_df.loc[mcf_mif_icf_df['Project Displayname'] == r'SPARC']['nTtauEstar_max'].iloc[0]
    # SPARC has rebaselined Q>1 to 2027
    sparc_minus_error = 4.1e21 # lower bound is at bottom of what's projected, Q_fuel = 2
    sparc_rect = patches.Rectangle((datetime(2027,1,1), sparc_tp-sparc_minus_error), timedelta(days=365*5), sparc_minus_error, edgecolor='white', facecolor='red', alpha=1, hatch='////')

    ax.add_patch(sparc_rect)
    annotation = {'text': 'SPARC',
                  'xy': (datetime(2029,7,1), sparc_tp - 2e21),
                  'xytext': (datetime(2025,1,1), 6e20),
                  'arrowprops': {'arrowstyle': '->'},
                  'zorder': 10,
                 }
    ax.annotate(**annotation)
    
    #ITER
    iter_tp = mcf_mif_icf_df.loc[mcf_mif_icf_df['Project Displayname'] == r'ITER']['nTtauEstar_max'].iloc[0]
    iter_minus_error = 2.2e21 # lower bound is at bottom of what's projected, Q_fuel = 10
    # ITER has rebaselined D-T operations to 2039.
    iter_rect = patches.Rectangle((datetime(2039,1,1), iter_tp - iter_minus_error), timedelta(days=365*5), iter_minus_error, 
                                 edgecolor='white', facecolor='red', alpha=1, hatch='////', linewidth=1, zorder=2)
    ax.add_patch(iter_rect)
    annotation = {'text': 'ITER',
                  'xy': (datetime(2041,7,1), iter_tp),
                  'xytext': (datetime(2039,1,1), 1e21),
                  'arrowprops': {'arrowstyle': '->'},
                  'zorder': 10,
                 }
    ax.annotate(**annotation)
    
    # Label horizontal Q_sci^MCF lines
    ax.annotate(r'$Q_{\rm sci}^{\rm MCF}=\infty$', (mcf_horizontal_range_dict[float('inf')][0]+timedelta(days=365*0.5), 1.22e22), alpha=1, color='red')
    ax.annotate(r'$Q_{\rm sci}^{\rm MCF}=10$', (mcf_horizontal_range_dict[10][0]+timedelta(days=365*0.5), 6.85e21), alpha=1, color='red')
    ax.annotate(r'$Q_{\rm sci}^{\rm MCF}=2$', (mcf_horizontal_range_dict[2][0]+timedelta(days=365*0.5), 2.55e21), alpha=1, color='red')
    ax.annotate(r'$Q_{\rm sci}^{\rm MCF}=1$', (mcf_horizontal_range_dict[1][0]+timedelta(days=365*0.5), 1.45e21), alpha=1, color='red')

    # Draw projection legend rectangle
    projection_rect = patches.Rectangle((datetime(1961,1,1), 1.5e12), timedelta(days=365*5), 2e12, edgecolor='white', facecolor='red', alpha=1, hatch='////', zorder=10)
    ax.add_patch(projection_rect)
    ax.annotate('Projections', xy=(datetime(1967,1,1), 1.7e12), xytext=(datetime(1967,1,1), 1.7e12), xycoords='data', alpha=1, color='black', size=10, zorder=10)

    # Caveat Q_sci_^MCF
    #ax.annotate(r'$Q_{\rm sci}^{\rm MCF}$' + r'assumes $T_i=15 {\rm keV}$', (1960, 1e22), color='red', size=9)

    # Annotate NIF Ignition Shots
    # Define the ellipse parameters
    #center_x, center_y = 2022, 5e21
    #width, height = 5, 0.4e22  # Width and height in data coordinates
    #ellipse = Ellipse((center_x, center_y), width, height, edgecolor='black', facecolor='none', transform=ax.transData)
    #ax.add_patch(ellipse)

    # Add watermark
    if add_prepublication_watermark:
        ax.annotate('Prepublication', (datetime(1960,1,1), 1.5e13), alpha=0.1, size=60, rotation=45)
    
    # Legend to the right
    #plt.legend(bbox_to_anchor=(1, 1.015), ncol=1)
    
    # Legend below
    #plt.legend(bbox_to_anchor=(1.01, -0.12), ncol=4)
    
    # Legend inside graph
    plt.legend(ncol=2)
    
    plt.show()
    fig.savefig(os.path.join('images', label_filename_dict['fig:scatterplot_nTtauE_vs_year']), bbox_inches='tight')


# %% [markdown]
# <HR><HR>

# %% [markdown]
# # Interactive Plotly Plots

# %% [markdown]
# Using [Plotly](https://plotly.com/python/interactive-html-export/), as it exports interactive HTML directly from Python and are easy to embed.

# %%
import plotly
import plotly.graph_objects as go
import plotly.colors as pc

# %% [markdown]
# Converting matplotlib notations for symbols to plotly convention.

# %%
concept_dict_conversion = {
    'o': 'circle',
    '*': 'star',
    'v': 'triangle-down',
    '|': 'hexagon-open',
    '2': 'triangle-down-open',
    'd': 'diamond',
    's': 'square',
    'X': 'x',
    '_': 'hexagram',
    'p': 'pentagon',
    'x': 'x',
    '.': 'circle-open'
}
icf_curve_conversion = {
    1 : 'dot',
    2 : 'dash'
}

 #changing to uniform marker shapes
for concept in concept_dict: concept_dict[concept]['markersize'] = 12


# %% [markdown]
# ### Plotly HTMLplot export function

# %%
def HTMLplot_to_html(fig, name = 'HTMLplot.html', include_libs = 'cdn', figID = 'my_plot'):
    '''Export plotly figures to HTML. Give custom links for each datapoints as 'data.points[0].customdata' '''
    fig.write_html(
        name,
        include_plotlyjs=include_libs,
        include_mathjax=include_libs,
        full_html=False,
        config={
            'scrollZoom': False,
            'displayModeBar': False
        },
        div_id=figID,
        post_script=f"""
            document.getElementById('{figID}').on('plotly_click', function(data) {{
                const url = data.points[0].customdata;
                if (url) {{
                    window.open(url, '_blank');
                }}
            }});
            """
    )


# %% [markdown]
# ### Plotly Triple Product vs Ion Temperature plot function

# %% jupyter={"source_hidden": true}
def plotlyplot_tripleprod_vs_T(add_prepublication_watermark= True):
    '''Returns Triple Product vs Ion Temperature plotly figures to HTML.'''
    
    fig = go.Figure()
    

    # Configure axes to be logarithmic and set ranges
    fig.update_xaxes(
        type="log",
        range=[np.log10(0.01), np.log10(100)],
        title = dict(
            text=r"$T_{i0}, \langle T_i \rangle_{\rm n}\;(\mathrm{keV})$",
            font=dict(size=18)
        ),
        showgrid=True, gridwidth=1, gridcolor="lightgrey",
        tickfont=dict(size=16),
        dtick=1
    )
    
    fig.update_yaxes(
        type="log",
        range=[np.log10(1e12), np.log10(1e23)],
        title = dict(
            text=r"$n_{i0} T_{i0} \tau_E^*,\;n\langle T_i\rangle_{\rm n}\tau_{\rm stag}\;(\mathrm{m^{-3}\,keV\,s})$",
            font=dict(size=18)
        ),
        tickfont=dict(size=16),
        showgrid=True, gridwidth=1, gridcolor="lightgrey",
    )
    
    
    
    # MCF filled bands
    for band in mcf_bands:
        Q = band['Q']
        y_low  = DT_requirements_df[f"{mcf_ex1.name}__nTtauE_Q_fuel={Q}"]
        y_high = DT_requirements_df[f"{mcf_ex2.name}__nTtauE_Q_fuel={Q}"]
    
        
        #1a) Lower boundary of trace
        fig.add_trace(go.Scatter(
            x=DT_requirements_df['T_i0'],
            y=y_low,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'      
        ))
        
        # 1b) Upper boundary of trace
        fig.add_trace(go.Scatter(
            x=DT_requirements_df['T_i0'],
            y=y_high,
            mode='lines',
            line=dict(width=0),
            showlegend=False,      
            hoverinfo='skip',
            
            fill='tonexty',       # fill down to previous y
            fillcolor="rgba(255,0,0,0.5)"  
        ))
    # Right‐side labels for MCF curve lines
    annotations = [
        (r'$Q_{\rm sci}^{\rm MCF}$',1.3e23),
        (r'_____', 8.4e22),
        (r'$\infty$', 3.2e22),
        (r'$10$', 1.8e22),
        (r'$2$', 8.5e21),
        (r'$1$', 4e21),
        (r'$0.1$', 6e20),
        (r'$0.01$', 5e19),
        (r'$0.001$', 6e18),
    ]
    
    # Add each annotation to the figure for MCF curve lines
    for text, y in annotations:
        fig.add_annotation(
            text=text,
            x=1.01,
            y=(math.log10(y) - fig.layout.yaxis.range[0])/(fig.layout.yaxis.range[1] - fig.layout.yaxis.range[0]),
            xanchor='left',
            yanchor='bottom',
            showarrow=False,
            font=dict(color='red'),
            xref="paper", yref="paper",
        )
        
    # ICF line curves
    for curve in icf_curves:
        Q = curve['Q']
        y = DT_requirements_df[f"{icf_ex.name}__nTtauE_Q_{q_type}={Q}"]
        
        fig.add_trace(go.Scatter(
            x=DT_requirements_df['T_i0'],
            y=y,
            mode='lines',
            line=dict(
                color=curve['color'],
                dash=icf_curve_conversion.get(curve['dashes'], 'solid'),   
                width=1
            ),
            opacity=curve['alpha']*0.8,
            showlegend=False,    
            hoverinfo='skip'
        ))
        
    # Inner annotations for ICF band
    fig.add_annotation(
        x=0.5, y=1,
        xref="paper", yref="paper",
        text=r"$(n T \tau_{\rm stag})_{\rm ig, hs}^{\rm ICF}$",
        font=dict(color="black"),
        showarrow=False
    )

    # Loop over each concept and add a Scatter trace
    for concept in concept_list:
        df = mcf_mif_icf_df[mcf_mif_icf_df['Concept Displayname'] == concept] 
    
        # Annotate the exceptions ('SPARC' & 'ITER')
        annotation = []
        for index, row in df.iterrows():
            displayname = row['Project Displayname']
            nTtauE_indicator = nTtauE_indicators.get(displayname, default_indicator)
            text = row['Project Displayname']
            if text in ['SPARC', 'ITER']:
                text += '*'
            annotation += [text]

        # Temparay solution, can be changed later 
        customlinks = ["https://www.fusionenergybase.com/"] * len(df['nTtauEstar_max'])
        
        fig.add_trace(go.Scatter(
            x=df['T_i_max'],
            y=df['nTtauEstar_max'],
            mode='markers',
            name=concept,
            marker=dict(
                color=concept_dict[concept]['color'],
                symbol=concept_dict_conversion.get(concept_dict[concept]['marker'], 'circle'),  
                size=concept_dict[concept]['markersize'],
                line=dict(
                    color='grey',
                    width=1
                )
            ),
            customdata=customlinks,
            hovertemplate=(
                "<b>%{text}</b><br>"
                r"T<sub>i</sub>: %{x} keV<br>"
                r"nTτ: %{y:.2e}<extra></extra>"
            ),
            text=[x+' : '+ y for x,y in zip(annotation, [concept] * len(df))]
        ))

    #Small text
    fig.add_annotation(
        x=1, y=0,
        xref="paper", yref="paper",
        text="* Maximum projected",
        font=dict(color="black", size=10),
        showarrow=False
    )
    
    # Prepublication watermark
    if add_prepublication_watermark:
        fig.add_annotation(
            text="Prepublication",
            x=0.5,          
            y=0.5,          
            xref="paper",  
            yref="paper",
            opacity=0.1,
            font=dict(size=70),
            textangle=-45,
            showarrow=False
        )    
    # Disable zoom interactions and general sanitation
    fig.update_layout(
        width=700,      
        height=700,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        legend_tracegroupgap=0,
        legend=dict(
            title="Concept",
            title_font=dict(size=12),      
            font=dict(size=10),            
            x=0.01,                        
            y=0.99,                        
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",  
            bordercolor="black",
            borderwidth=1
        ),
        title=dict(
            text="Triple Product VS Ion Temperature",
            x=0.5,               
            xanchor="center",
            y=0.0,              
            yanchor="bottom",
        ),
        margin=dict(b=80)
    )
    
    return fig
    
HTMLplot_to_html(plotlyplot_tripleprod_vs_T())
