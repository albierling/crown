import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

import seaborn as sns
from wordcloud import WordCloud
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
from PIL import Image
from rdkit.Chem import Descriptors
from collections import Counter
import os
from os import path
import subprocess
import requests

# Set the page layout to 'wide'
st.set_page_config(layout="wide")

# Define file paths
CROWN_FILE = "data.xlsx"
ODORS_FILE = "odors.xlsx"
ODORS_EXTENDED_FILE = "odors_extended.xlsx"
ZENODO_ID = "14727277"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/albierling/crown/main/" 

# Load the datasets using the new caching method
@st.cache_data
def load_data():
   # Ensure odors_extended.xlsx is available (from GitHub)
   if not os.path.isfile(ODORS_EXTENDED_FILE):
      st.write(f"Downloading {ODORS_EXTENDED_FILE} from GitHub...")
      file_url = GITHUB_RAW_URL + ODORS_EXTENDED_FILE
      try:
         response = requests.get(file_url)
         if response.status_code == 200:
             with open(ODORS_EXTENDED_FILE, "wb") as f:
                 f.write(response.content)
             st.write(f"✅ Successfully downloaded {ODORS_EXTENDED_FILE}.")
         else:
             st.error(f"❌ Failed to download {ODORS_EXTENDED_FILE} from GitHub (Status: {response.status_code})")
      except Exception as e:
         st.error(f"❌ GitHub download failed: {e}")
   
   # Download from Zenodo if necessary
   if not os.path.isfile(ODORS_FILE) or not os.path.isfile(CROWN_FILE):
     placeholder = st.empty()
     placeholder.write("Downloading datasets from Zenodo...")
     subprocess.run(f"zenodo_get {ZENODO_ID}", shell=True, text=True, check=True)
     placeholder.empty()
   
   # Load datasets
   odors_extended_data = pd.read_excel(ODORS_EXTENDED_FILE)  # From GitHub
   crown_data = pd.read_excel(CROWN_FILE)  # From Zenodo
   odors_data = pd.read_excel(ODORS_FILE)  # From Zenodo
   
   # Merge the datasets on 'molcode', keeping all rows from odors_data
   merged_odors_data = pd.merge(odors_data, odors_extended_data, on='molcode', how='left')
   return crown_data, merged_odors_data

#def select_font(language, display_fontfile=False):
#   # rather hacky way to select the right font
#   noto = 'NotoSans-Regular'
#   if language == 'Chinese':
#     noto = 'NotoSansCJK-Regular'
#   elif language == 'Hebrew':
#     noto = 'NotoSansHebrew-Regular'
#   elif language == 'Hindi':
#     noto = 'NotoSansDevanagari-Regular'
   
#   flist = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
#   fn_noto = ''
#   for fn in flist:
#     if noto in fn:
#         fn_noto = fn
#         break
   
#   ## select font for word cloud
#   try:
#     font_file = font_manager.findfont('Arial Unicode MS', fallback_to_default=False)
#   except:
#     font_search = font_manager.FontProperties(fname=fn_noto)
#     font_file = font_manager.findfont(font_search)
#   
#   if display_fontfile:
#     st.write('Font: ' + font_file)
#   
#   return font_file

# Load data
original_data, merged_odors_data = load_data()
crown_data = original_data.copy()
crown_data = crown_data[crown_data["odor_set"] != 10] # remove patients from dataset here

# Function to compute molecular properties
@st.cache_data
def calculate_properties_cached(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    molecular_weight = Descriptors.ExactMolWt(molecule)
    size = molecule.GetNumAtoms()
    return molecular_weight, size

# Get the most frequent free descriptions for the selected molecule
def get_top_descriptions(crown_data, selected_molcode, top_n=25):
    filtered_data = crown_data[crown_data['molcode'].isin([selected_molcode])]
    filtered_data['parsed_descriptions'] = filtered_data['free_description'].apply(parse_free_descriptions)
    all_descriptions = [desc for sublist in filtered_data['parsed_descriptions'] for desc in sublist]
    description_counts = Counter(all_descriptions)
    top_descriptions = description_counts.most_common(top_n)
    return top_descriptions

# Function to filter dataset based on selected descriptions
def filter_by_description(crown_data, selected_molcode, selected_descriptions):
    filtered_data = crown_data[crown_data['molcode'].isin([selected_molcode])]
    filtered_data['parsed_descriptions'] = filtered_data['free_description'].apply(parse_free_descriptions)
    filtered_data = filtered_data[filtered_data['parsed_descriptions'].apply(lambda x: any(desc in x for desc in selected_descriptions))]
    return filtered_data

# Function to safely convert free descriptions to a list
def parse_free_descriptions(description):
    if isinstance(description, str):
        try:
            return eval(description)
        except:
            return []
    elif isinstance(description, list):
        return description
    else:
        return []

# Function to generate word cloud from free descriptions
@st.cache_data
def generate_word_cloud_cached(free_descriptions, word_limit, colormap):
    words = [word for sublist in free_descriptions for word in sublist]
    words = sum(free_descriptions, [])
    text = ' '.join(words)

    w, c = np.unique(words, return_counts=True)
    freq_dict = dict(zip(list(w), list(c)))

    d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    mask = np.array(Image.open(path.join(d, 'mask.png')))

    #font_file = select_font('English')
    wordcloud = WordCloud(width=800, height=400, max_words=word_limit,
                          font_path='NotoSans_Condensed-SemiBold.ttf',
                          mask=mask, 
                          background_color='white', 
                          colormap=colormap,
                          random_state=42).generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Define a mapping of English dimension names to German labels
dimension_labels = {
    'pleasant': 'angenehm',
    'intensive': 'intensiv',
    'familiar': 'vertraut',
    'edible': 'essbar',
    'disgusting': 'ekelerregend',
    'warm': 'warm',
    'cold': 'kalt',
    'irritating': 'reizend'
}

# Function to plot the pleasantness and intensity distributions without grid
def plot_distributions(mol_data, selected_dimensions, title_prefix=""):
    num_plots = len(selected_dimensions)
    num_cols = 4
    num_rows = (num_plots + num_cols - 1) // num_cols
    
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(16, 4 * num_rows))
    ax = ax.flatten()  # Flatten in case the number of dimensions is less than 8
    
    sns.set_palette("bright")
    sns.set_style("dark")

    colors = ["indianred", "salmon", "tomato", "lightcoral", 
              "darksalmon", "orchid", "crimson", "pink"]
    
    for i, dimension in enumerate(selected_dimensions):
        # Map the English dimension name to its German label for the title and legend
        german_label = dimension_labels[dimension]
        
        sns.histplot(mol_data[dimension], bins=20, ax=ax[i], color=colors[i % len(colors)], kde=True)
        ax[i].set_xlabel("1 = gar nicht bis 100 = sehr", fontsize=12)  # Update x-axis label to German
        ax[i].set_ylabel("Anzahl der Bewertungen", fontsize=12)  # Update y-axis label to German
        ax[i].set_title(f"{title_prefix} {german_label.capitalize()}", fontsize=16, fontweight="bold")  # Use German label for title
        #ax[i].legend([german_label])  # Use German label for legend
        
        # Turn off the grid
        ax[i].grid(False)
        
    for i in range(num_plots, num_rows * num_cols):
        ax[i].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)


######################################################################################
######################################################################################

# App Title
st.title("Entwicklung eines chemisch-perzeptuellen Raums olfaktorischer Wahrnehmung (CROWN)")
st.write("Warum riecht etwas angenehm oder intensiv und wieso unterscheidet sich die Wahrnehmung von Gerüchen zwischen Menschen?" 
         "  Mit der CROWN-Studie untersuchen wir, wie chemische Eigenschaften von Duftmolekülen mit Dimensionen der Wahrnehmung zusammenhängen." 
         " In dieser Datenbank kann der Datensatz der CROWN-Studie interaktiv exploriert werden.")

with st.expander("**Hintergrund zur Studie**"):
    st.write("Ob und wie Menschen Gerüche wahrnehmen, hängt davon ab, wie diese chemisch aufgebaut sind." 
        " Es ist beispielsweise bekannt, dass Moleküle nur dann überhaupt nach etwas riechen, wenn" 
        " sie eine bestimmte Löslichkeit in Fett und Wasser aufweisen, damit sie die Nasenschleimhaut" 
        " passieren und an Rezeptoren des Riechsystems binden können. Außerdem wissen wir bereits," 
        " dass ganz bestimmte chemische Strukturen dafür verantwortlich sind, ob wir z.B. einen blumigen," 
        " fruchtigen oder ranzigen Geruch wahrnehmen. Wie genau die Struktur des Moleküls die Wahrnehmung" 
        " von Gerüchen bestimmt, ist jedoch bislang kaum verstanden."          
        " Um diesem Zusammenhang zwischen Struktur und Wahrnehmung auf den Grund zu gehen, ist es daher nötig," 
        " den Geruch unterschiedlich aufgebauter Moleküle mit verschiedenen Wahrnehmungsdimensionen wie Angenehmheit," 
        " Intensität oder Vertrautheit zu bewerten. Das ist Zielstellung der CROWN-Studie, kurz für 'Entwicklung eines"
        " chemisch-perzeptuellen Raums olfaktorischer Wahrnehmung'.")
    st.write("Ziel dieser Datenbank ist es, eine wichtige Wissenslücke zu schließen und möglichst viele Daten für" 
        " die Geruchswahrnehmung verschiedener Molekülen zu sammeln und untersuchen. Zu diesem Zweck haben in der CROWN-Studie" 
        " über 1000 Personen an einer Auswahl von 10 aus insgesamt 74 mono-molekularen Geruchsstoffen geschnuppert"
        " und diese sowohl frei beschrieben als auch nach Maßen wie Angenehmheit und Intensität bewertet. In dieser" 
        " Webanwendung können Sie einen Teil der Ergebnisse der CROWN-Studie selbst erforschen, z.B., wie angenehm oder intensiv ein" 
        " Geruch wahrgenommen wird und wie sehr die Meinungen dabei auseinandergehen. Indem Sie ein Molekül" 
        " auswählen, können Sie sich z.B. die Bewertung von Angenehmheit, Intensität und Vertrautheit anzeigen lassen"
        " und in einer Wortwolke die häufigsten freien Beschreibungen anzeigen.")    
  

######################################################################################
############# Sidebar ################################################################
# Sidebar for molecule selection and description buttons
with st.sidebar:
   st.image('crown-logo.png', width=300)

   molcodes = crown_data['molcode'].unique()
   sampling_groups = crown_data['sampling_group'].unique()
    
   #############################
   st.subheader("Auswahl von Molekül und Filtern")
   st.write("Hier kann das Molekül sowie die untersuchten Testgruppen ausgewählt werden.")
   ##############################
   col1, col2 = st.columns([1, 1])            
   
   with col1:
     selected_molcode = st.sidebar.selectbox('Ausgewähltes Molekül', molcodes, index=2)
   
   with col2:
     # Get the list of German labels for selection
     german_dimensions = list(dimension_labels.values())
   
     # Use the German labels for display in the multiselect
     selected_german_dimensions = st.sidebar.multiselect('Ausgewählte Bewertungsdimensionen', german_dimensions, default=german_dimensions[:4])
   
     # Map the selected German labels back to their corresponding English dimension keys
     selected_dimensions = [key for key, value in dimension_labels.items() if value in selected_german_dimensions]
     #dimensions = ['pleasant', 'intensive', 'familiar', 'edible', 'disgusting', 'warm', 'cold', 'irritating']
     #selected_dimensions = st.multiselect('Ausgewählte Bewertungsdimensionen', dimensions, default=dimensions[:4])
   
   ########################
   st.subheader("Anpassung der Wortwolke")
   # Define columns
   col1, col2 = st.columns([1, 1])
   
   max_no = 1500
   with col1: 
     word_limit = st.slider("Maximale Wörter, die in die Wortwolke eingehen", min_value=10, max_value=max_no, value=100)
   with col2:
     colormap = st.selectbox("Farbschema für die Wortwolke", ["plasma", "viridis", "magma", "inferno", "Blues"])
   
   # Simulate 'anchoring' by placing content at the end of the sidebar block
   st.markdown("---")  # Horizontal line to separate content
   
   ###### Logos etc.
   st.write("Diese Forschung wurde finanziert vom Projekt 'Olfactorial Perceptronics', gefördert von der " 
          " VolkswagenStiftung. Mehr Informationen zum Projekt siehe https://perceptronics.science/.")
   
   # Simulate 'anchoring' by placing content at the end of the sidebar block
   st.markdown("---")  # Horizontal line to separate content
   
   if os.path.isfile('logos-tud-fsu-vws.png'):
      st.image('perceptronics-logo-blau.png', width=170)
      st.image('logos-tud-fsu-vws.png', width=300)

############# Sidebar end ############################################################


############# Main #####################################################################

mol_data = crown_data[crown_data['molcode'] == selected_molcode]
mol_data = mol_data[mol_data['inclusion'] == 1]

german_name = merged_odors_data.loc[merged_odors_data['molcode'] == selected_molcode, 'german_name'].values[0]
sampling_groups_count = mol_data['sampling_group'].value_counts()

sociodemographic_data = mol_data
sample_size = len(sociodemographic_data['code'].unique())
mean_age = round(np.mean(sociodemographic_data['age']), 2)
std_age = round(np.std(sociodemographic_data['age']), 2)

mean_pleasantness = mol_data['pleasant'].mean()
mean_intensity = mol_data['intensive'].mean()

top_descriptions = get_top_descriptions(mol_data, selected_molcode)
########################################################

st.subheader("Chemische Eigenschaften")
german_name = merged_odors_data.loc[merged_odors_data['molcode'] == selected_molcode, 'german_name'].values[0]
st.write(f"#####  {german_name}")

col1, col2 = st.columns([1, 3])

with col1:
   smiles = merged_odors_data.loc[merged_odors_data['molcode'] == selected_molcode, 'SMILES'].values[0]
   mol_structure = Chem.MolFromSmiles(smiles)

   if mol_structure:
      mol_image = Draw.MolToImage(mol_structure, size=(200, 200))
      st.image(mol_image)
   else:
      st.error(f"Failed to generate molecule structure for {german_name}. Please check your input.")


with col2:
        
        
   # Display the 'smell' and 'occurrence' information from the extended file
   smell_description = merged_odors_data.loc[merged_odors_data['molcode'] == selected_molcode, 'smell'].values[0]
   occurence = merged_odors_data.loc[merged_odors_data['molcode'] == selected_molcode, 'occurence'].values[0]
   cite1 = merged_odors_data.loc[merged_odors_data['molcode'] == selected_molcode, 'cite1'].values[0]
   cite2 = merged_odors_data.loc[merged_odors_data['molcode'] == selected_molcode, 'cite2'].values[0]
   cite3 = merged_odors_data.loc[merged_odors_data['molcode'] == selected_molcode, 'cite3'].values[0]

   # Only display if the value is not NaN
   if pd.notna(smell_description):
      st.markdown(f"""
      <div style='padding:10px; border: 3px solid #4CAF50; border-radius: 10px; background-color: #eaffea;'>
      <strong>Geruchsbeschreibung in Parfüm-/Chemiedatenbanken:</strong> {smell_description}
      </div>
      """, unsafe_allow_html=True)

   if pd.notna(occurence):
      st.write(f"**Vorkommen und Anwendung:** {occurence}")

      # Display molecular weight and size
      molecular_weight, size = calculate_properties_cached(smiles)
      st.write(f"**Molekulargewicht:** {molecular_weight:.2f} g/mol")
      st.write(f"**Größe (Anzahl an Atomen):** {size}")

      # Collect all citations that are not NaN
      citations = [cite for cite in [cite1, cite2, cite3] if pd.notna(cite)]

   if citations:
      # Join the citations with commas and display them
      st.caption(f":globe_with_meridians: {', '.join(citations)}")

        

st.subheader("Bewertungsdimensionen auf visuellen Analogskalen")
sentence = ("Im Folgenden können bis zu acht Bewertungsdimensionen angeschaut werden. Die Abbildungen zeigen je die Verteilung der"
                " Bewertungen auf der Skala von 1 = gar nicht bis 100 = sehr. Für manche Gerüche ist dabei die Verteilung sehr breit, d.h.,"
                " die Probandinnen und Probanden sind sich eher uneinig, ob der Geruch z.B. angenehm oder intensiv ist. Für andere Gerüche"
                " gibt es klarere Tendenzen, z.B. häufig eine stark linkssteile Verteilung für 'essbar'."
                f" Das Molekül mit dem offiziellen Namen **{german_name}** wurde von insgesamt **{sample_size}** Personen bewertet. "
                f"{', '.join([f'{v} aus der {k} Gruppe' for k, v in sampling_groups_count.items()])}. "
                f"Die Stichprobe war im Mittel {mean_age} Jahre alt mit einer Streuung von {std_age} Jahren."
                f" Der Geruch wurde im Mittel als **{mean_pleasantness:.2f} angenehm** und **{mean_intensity:.2f} intensiv** bewertet.")
st.write(sentence)

sns.set_style("dark")
plot_distributions(mol_data, selected_dimensions)

################################
st.subheader("Freie Beschreibungen")
st.write("Im folgenden sind die 25 häufigsten freien Beschreibungen für den ausgewählten Geruch aufgelistet (Abbildung unten links)." 
                " Die rechte untere Abbildung zeigt eine Wortwolke, die die Häufigkeiten der freien Beschreibungen anhand der Größe des jeweiligen" 
                " Wortes darstellt. Um die Anzahl der zu berücksichtigenden Wörter anzupassen, kann links in der Seitenleiste zwischen mindestens 10"
                " und maximal 1500 Wörtern ausgewählt werden. Die freien Beschreibungen der Probandinnen und Probanden wurden für beide Abbildungen"
                " zuvor standardisiert, d.h., zum Beispiel verschiedene Schreibweisen der gleichen Beschreibung wie z.B. 'süßlich' und 'süß' zusammengefasst."
                " Dadurch wurden aus insgesamt mehr als 13000 freien Beschreibungen eine Liste von ca. 3500 einzigartigen Beschreibungen - je nach Geruch"
                " kommen unterschiedlich viele davon vor.")

col1, col2 = st.columns([1, 3])            

sns.set_style("whitegrid")
with col1:
   top_descriptors = get_top_descriptions(crown_data, selected_molcode)
   df_descriptors = pd.DataFrame(top_descriptors, columns=['Descriptor', 'Frequency'])
   df_descriptors = df_descriptors.sort_values(by='Frequency', ascending=False)

   sns.set_style("whitegrid")
   fig, ax = plt.subplots(figsize=(4, 10))
   sns.barplot(x='Frequency', y='Descriptor', data=df_descriptors, palette='magma')

   ax.set_xlabel('')
   ax.set_ylabel('')
   ax.tick_params(axis='y', labelsize=12)
   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)
   ax.spines['left'].set_visible(False)
   ax.spines['bottom'].set_visible(False)
   ax.axes.xaxis.set_visible(False)
   ax.grid(False)

   for index, value in enumerate(df_descriptors['Frequency']):
      ax.text(value, index, f'{value}', va='baseline', ha='left', fontsize=12)

   st.pyplot(fig)

with col2:
   free_descriptions = mol_data['free_description'].apply(parse_free_descriptions)
   wordcloud_fig = generate_word_cloud_cached(free_descriptions, word_limit, colormap)
   st.pyplot(wordcloud_fig)
