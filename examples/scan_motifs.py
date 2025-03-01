# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
import os
import math
import h5py
import numpy as np
import pandas as pd

sys.path.append('/data/scratch/rdeng/enhancer_project/ipython_notebooks/')
from helper import IOHelper, SequenceHelper 
from deeplift.visualization import viz_sequence


# %% [markdown]
# #### Prepare files for TF-modisco-lite

# %%
# threshold

# category5 -> category_5
NSC_max = 7.73604582550066 # minimum: 1.40219933979478, maximum: 7.73604582550066
ESC_max = 5.70545787726145 # minimum: 1.48829212312168, maximum: 5.70545787726145

NSC_category5 = 1.40219933979478 # minimum: 1.40219933979478, maximum: 7.73604582550066
ESC_category5 = 1.48829212312168 # minimum: 1.48829212312168, maximum: 5.70545787726145

# category4 -> category_4
NSC_category4 = 1.14848354443742 # minimum: 1.14848354443742, maximum: 1.40219325414005
ESC_category4 = 1.15041322844751 # minimum: 1.15041322844751, maximum: 1.48825248552868

# category3 -> category_3
NSC_category3 = 0.972399595786853 # minimum: 0.972399595786853, maximum: 1.14847961165673
ESC_category3 = 0.934903719163151 # minimum: 0.934903719163151, maximum: 1.15039662982546

# category2 -> category_2
NSC_category2 = 0.791864633976558 # minimum: 0.791864633976558, maximum: 0.972393932644748
ESC_category2 = 0.740407828029512 # minimum: 0.740407828029512, maximum: 0.9348999542536

# category1 -> category_1
NSC_category1 = 0 # minimum: 0, maximum: 0.791863939840448
ESC_category1 = 0 # minimum: 0, maximum: 0.740404902305098  

# load the contribution scores
contri_NSC = np.load('/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/Whole/shap_explanations_NSC.npy')
contri_ESC = np.load('/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/Whole/shap_explanations_ESC.npy')
# acutually inp_NSC and inp_ESC are same
inp_NSC = np.load('/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/Whole/inp_NSC.npy')
inp_ESC = np.load('/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/Whole/inp_ESC.npy')

# load initial dataset to filter for NSC-high, ESC-high and common-high
file_seq = str("/data/scratch/rdeng/enhancer_project/data/Enhancer.fa")
input_fasta = IOHelper.get_fastas_from_file(file_seq, uppercase=True)  

Activity = pd.read_table("/data/scratch/rdeng/enhancer_project/data/Enhancer_activity.txt")

NSC_category5_idx = np.where(np.logical_and(Activity.NSC_log2_enrichment >= NSC_category5, Activity.NSC_log2_enrichment <= NSC_max))[0]
ESC_category5_idx = np.where(np.logical_and(Activity.ESC_log2_enrichment >= ESC_category5, Activity.ESC_log2_enrichment <= ESC_max))[0]

# select contribution scores
contri_NSC_category5 = contri_NSC[NSC_category5_idx]
contri_ESC_category5 = contri_ESC[ESC_category5_idx]

inp_NSC_category5 = inp_NSC[NSC_category5_idx]
inp_ESC_category5 = inp_ESC[ESC_category5_idx]


# save 
# np.save('/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/5_categories/Category_5/shap_explanations_NSC_category5.npy', contri_NSC_category5)
# np.save('/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/5_categories/Category_5/shap_explanations_ESC_category5.npy', contri_ESC_category5)

# np.save('/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/5_categories/Category_5/inp_NSC_category5.npy', inp_NSC_category5)
# np.save('/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/5_categories/Category_5/inp_ESC_category5.npy', inp_ESC_category5)


# %%
# the motif setting is 5000 
# !mkdir -p /data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/5_categories/Category_5/h5Totable/

# %% [markdown]
# #### Make tables that contains enhancer ID, motif, motif_start and motif_end

# %%
# 6 different results: NSC_high, common_Nhigh, NSC_high_ESCmodel, ESC_high, common_Ehigh, ESC_high_NSCmodel
# we need to concatenate: enhancer_Id|patterns|motifs|motif_start|motif_end
# motif_start is shifted 250 bp

def motif_H5ToTable(h5, fasta_group):
    """
    h5: motifs h5 file
    fasta_group: dataframe with the enhancer ID
    concatenate motifs in h5 with input_fasta with enhancer id 
    """
    # retrive the information from the h5 and shape it into table
    data = []
    for pattern_type in ['pos_patterns', 'neg_patterns']:
        if pattern_type not in h5: # some h5 only has neg_patterns
            continue
        patterns = h5[pattern_type]
        pattern_names = list(patterns.keys())
        for pattern_name in pattern_names:
            seqlets = patterns[pattern_name]['seqlets']
            example_idx = seqlets['example_idx'][()] # extract example_idx information
            start = seqlets['start'][()]  # extract start information
            end = seqlets['end'][()]  # extract end information
            for idx, s, e in zip(example_idx, start, end):
                data.append((pattern_type, pattern_name, idx, s, e))  # append start and end to the data list
    df = pd.DataFrame(data, columns=['pattern_type', 'pattern_name', 'example_idx', 'motif_start', 'motif_end'])
    # the start and end shifted 250bp
    df['motif_start'] = df['motif_start'] + 250
    df['motif_end'] = df['motif_end'] + 250
    # join with enhancer ID
    merge_df = pd.merge(fasta_group, df, right_on='example_idx', left_index=True, how='left')
    merge_df.drop(['sequence', 'example_idx'], axis=1, inplace=True)
    
    return merge_df

def motif_location(table):
    """
    table: the output of motif_H5ToTable
    add a new column: motif location
    """
    new_table = table.copy()  # Make a copy of the input table to avoid modifying it directly
    new_coordinates = []
    for coordinate, motif_start in zip(new_table['location'], new_table['motif_start']):
        chromosome, interval = coordinate.split(':')
        start, end = interval.split('-')
        
        midpoint = (int(start) + int(end)) // 2
        new_start = midpoint - 500 + motif_start  # Move the start position left by 500 base pairs, add the value of "extension_start", but not past the beginning of the chromosome
        new_end = new_start + 30  # Move the end position right by 500 base pairs and add the value of "extension_start"
        

        if math.isnan(new_start):
            new_coordinate = float('nan')
        else:
            new_coordinate = f"{chromosome}:{int(new_start)}-{int(new_end)}"
        new_coordinates.append(new_coordinate)
    new_table['motif_location'] = new_coordinates
    new_table.drop(['motif_start', 'motif_end'], axis=1, inplace=True)
    
    return new_table

dic = {'NSC_category5': NSC_category5_idx, 
       'ESC_category5': ESC_category5_idx}

writer = pd.ExcelWriter("/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/5_categories/Category_5/h5Totable/enhancer_motif.xlsx") # Arbitrary output name
for group_name, idx_name in dic.items():
    fasta_group = input_fasta[input_fasta.index.isin(idx_name)].reset_index(drop=True)
    filename = str("/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/5_categories/Category_5/" + group_name + "/out_" + group_name + ".h5")
    h5 = h5py.File(filename, "r")

    h5_table = motif_H5ToTable(h5, fasta_group)
    motif_table = motif_location(h5_table)
    

    motif_table.to_excel(writer, sheet_name=os.path.splitext(group_name)[0], index=False)
writer.save()





# %%
def motif_H5ToTable(h5, fasta_group):
    """
    h5: motifs h5 file
    fasta_group: dataframe with the enhancer ID
    concatenate motifs in h5 with input_fasta with enhancer id 
    """
    # retrive the information from the h5 and shape it into table
    data = []
    for pattern_type in ['pos_patterns', 'neg_patterns']:
        if pattern_type not in h5: # some h5 only has neg_patterns
            continue
        patterns = h5[pattern_type]
        pattern_names = list(patterns.keys())
        for pattern_name in pattern_names:
            seqlets = patterns[pattern_name]['seqlets']
            example_idx = seqlets['example_idx'][()] # extract example_idx information
            start = seqlets['start'][()]  # extract start information
            end = seqlets['end'][()]  # extract end information
            for idx, s, e in zip(example_idx, start, end):
                data.append((pattern_type, pattern_name, idx, s, e))  # append start and end to the data list
    df = pd.DataFrame(data, columns=['pattern_type', 'pattern_name', 'example_idx', 'motif_start', 'motif_end'])
    
    # the start and end shifted 250bp
    df['motif_start'] = df['motif_start'] + 250
    df['motif_end'] = df['motif_end'] + 250
    # join with enhancer ID
    merge_df = pd.merge(fasta_group, df, right_on='example_idx', left_index=True, how='left')
    merge_df.drop(['sequence', 'example_idx'], axis=1, inplace=True)
    
    return merge_df


fasta_group = input_fasta[input_fasta.index.isin(NSC_high_idx)].reset_index(drop=True)
filename = str("/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/5_categories/Category_5/5000/NSC_high/out_NSC_high.h5")
h5 = h5py.File(filename, "r")

motif_table = motif_H5ToTable(h5, fasta_group)

def motif_location(table):
    """
    table: the output of motif_H5ToTable
    add motif location
    """
    new_table = table.copy()  # Make a copy of the input table to avoid modifying it directly
    
    # some enhancers are padded with 0, then need to adjust the coordinates
    new_coordinates = []
    for coordinate, motif_start in zip(new_table['location'], new_table['motif_start']):
        chromosome, interval = coordinate.split(':')
        start, end = interval.split('-')
        
        midpoint = (int(start) + int(end)) // 2
        new_start = midpoint - 500 + motif_start  # Move the start position left by 500 base pairs, add the value of "extension_start", but not past the beginning of the chromosome
        new_end = new_start + 30  # Move the end position right by 500 base pairs and add the value of "extension_start"
        
        # some enhancers don't have motifs and are marked with nan
        if math.isnan(new_start):
            new_coordinate = float('nan')
        else:
            new_coordinate = f"{chromosome}:{int(new_start)}-{int(new_end)}"
        new_coordinates.append(new_coordinate)
    new_table['motif_location'] = new_coordinates
    new_table.drop(['motif_start', 'motif_end'], axis=1, inplace=True)
    
    return new_table

motif_location(motif_table)


tomtom_result = pd.read_table("/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/NSC_ESC_common/5000/common_Ehigh/tomtom_results.txt", header=None)


# %% [markdown]
# #### Visualize motifs

# %%
# threshold
NSC_top10 = 1.65208463524962
ESC_top10 = 1.82043324245546

# load contribution score from Test data
contri_scores = np.load('/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/NSC_ESC_common/shap_explanations_ESC_high.npy')
inps = np.load('/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/NSC_ESC_common/inp_ESC_high.npy')

# load initial dataset to filter for NSC-high, ESC-high and common-high
file_seq = str("/data/scratch/rdeng/enhancer_project/data/Enhancer.fa")
input_fasta = IOHelper.get_fastas_from_file(file_seq, uppercase=True)  

Activity = pd.read_table("/data/scratch/rdeng/enhancer_project/data/Enhancer_activity.txt")

NSC_high_idx = np.where(np.logical_and(Activity.NSC_log2_enrichment >= NSC_top10, Activity.ESC_log2_enrichment < ESC_top10))[0]
ESC_high_idx = np.where(np.logical_and(Activity.NSC_log2_enrichment < NSC_top10, Activity.ESC_log2_enrichment >= ESC_top10))[0]
common_high_idx = np.where(np.logical_and(Activity.NSC_log2_enrichment >= NSC_top10, Activity.ESC_log2_enrichment >= ESC_top10))[0]

NSC_high = input_fasta[input_fasta.index.isin(NSC_high_idx)].reset_index(drop=True)
ESC_high = input_fasta[input_fasta.index.isin(ESC_high_idx)].reset_index(drop=True)
common_high = input_fasta[input_fasta.index.isin(common_high_idx)].reset_index(drop=True)


# %%
# NSC-high
NSC_high.loc[NSC_high['location'] == 'chr8:123479279-123480279']
NSC_high.iloc[[3978]]
# test = NSC_high[NSC_high.index.isin(motif_idx1)]
# test.loc[test['location'] == 'chr8:123479279-123480279']

# ESC-high
ESC_high.loc[ESC_high['location'] == 'chr2:121490069-121490672']
ESC_high.iloc[[3451]]
# test = ESC_high[ESC_high.index.isin(motif_idx1)]
# test.loc[test['location'] == 'chr2:121490069-121490672']

# %%
import h5py
from deeplift.visualization import viz_sequence

filename = "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/NSC_ESC_common/5000/NSC_high/out_NSC_high.h5"
f = h5py.File(filename, "r")

# in total three layers of the h5
print(list(f['neg_patterns']))
print(list(f['neg_patterns']['pattern_0']))
print(list(f['neg_patterns']['pattern_0']['seqlets']))
# np.intersect1d(motif_idx1, motif_idx2)

motif_idx1 = f['neg_patterns']['pattern_0']['seqlets']['example_idx'][()]
motif_start1 = f['neg_patterns']['pattern_0']['seqlets']['start'][()]
motif_end1 = f['neg_patterns']['pattern_0']['seqlets']['end'][()]
print(motif_idx1)
print(motif_start1)
print(motif_end1)

# np.where(motif_idx1 == 3451)[0][0]
# motif_start1[np.where(motif_idx1 == 3451)[0][0]]

# %%
# 7375 is the previous one that I plotted with TP53 and YY2 for NSC-high
mod_viz = np.multiply(contri_scores[7375], inps[7375])
viz_sequence.plot_weights(mod_viz[:,:], subticks_frequency=20)


# %%
# 3079 TP53 for ESC-high
# mod_viz = np.multiply(contri_scores[3079], inps[3079])
# viz_sequence.plot_weights(mod_viz[:,440:480], subticks_frequency=20)

# 3451: the previous selected regions

# all plus 250
# pos_1 -> 45
# neg_2 -> 273, neg_7 -> 357, neg_12 -> 226


mod_viz = np.multiply(contri_scores[3451], inps[3451])
viz_sequence.plot_weights(mod_viz[:,:], subticks_frequency=20)

# %% [markdown]
# #### Check the motifs sites of selected HPO related NSC-high enhancers

# %%
select_enhancer = pd.read_csv("/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/NSC_ESC_common/NSC_high_selectedCoordinates.csv")

file_seq = str("/data/scratch/rdeng/enhancer_project/data/Enhancer.fa")
input_fasta = IOHelper.get_fastas_from_file(file_seq, uppercase=True)  

# load the contribution scores
contri_NSC = np.load('/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/Whole/shap_explanations_NSC.npy')
inp_NSC = np.load('/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/Whole/inp_NSC.npy')



# %%
select_fasta = input_fasta[input_fasta["location"].isin(select_enhancer["location"])]

select_idx = input_fasta[input_fasta["location"].isin(select_enhancer["location"])].index.tolist()
select_contri = contri_NSC[select_idx]
select_inp = inp_NSC[select_idx]

mod_viz = np.multiply(select_contri, select_inp)

for i in range(mod_viz.shape[0]):
    print(select_fasta.iloc[i,0])
    viz_sequence.plot_weights(mod_viz[i,:,:], subticks_frequency=20)
    print("")
