import pandas as pd
import numpy as np
import os



def B2_convert(site1, site2):
    assert type(site1) == str and type(site2) == str
    assert len(site1) == 1 and len(site2) == 1
    B3_dict = {'O':1, 'A':2, 'B':3}
    if site1 == site2:
        return site1*2
    else:
        site1_order = B3_dict[site1]; site2_order = B3_dict[site2];
        if site2_order < site1_order:
            site1, site2 = site2, site1
        return site1+site2+"+"+site2+site1

def B3_convert(B3):
    # B3 is a 3 letter string representing the 3-site correlation term
    assert type(B3) == str and len(B3) == 3
    B3_dict = {'O':1, 'A':2, 'B':3}
    if B3[0] == B3[2]:
        return B3
    else:
        return B3 + "+" + B3[::-1]


def load_data(indir, csv_name):
    # k_run.csv in ~/work/tafel/KMC was generated with indir = "/home/nricke/work/tafel/KMC/LV/k_run"
    #indir = "/home/nricke/work/tafel/KMC/500x_rep/" would generate for only a single set k's
    label_string = "t,A,B,OO,OA+AO,OB+BO,AA,AB+BA,BB,OOO,OOA+AOO,OOB+BOO,AOA,AOB+BOA,BOB,OAO,OAA+AAO,OAB+BAO,AAA,AAB+BAA,BAB,OBO,OBA+ABO,OBB+BBO,ABA,ABB+BBA,BBB"
    labels = label_string.split(',')
    print(labels)

    indir = "/home/nricke/work/tafel/KMC/LV/k_run"

    df_list = []
    #for infile in os.listdir(indir):
    for path, subdirs, files in os.walk(indir):
        print(path, subdirs, files)
        for name in files:
            if name[:3] == "run" and name[-3:] == "csv":
                df_list.append(pd.read_csv(path+'/'+name, header=None, names=labels))

    print(len(df_list))
    df = pd.concat(df_list)
    df.to_csv(csv_name)


def prep_df(infile, B3_string):
    ep = 10.**-10
    df = pd.read_csv(infile)

    assert len(B3_string) == 3 #this model is only made to predict 3-site terms

    """
    construct terms for B3_Estimate function that Andrew has been using
    map A,A to [AA], A,B and B,A to [AB+BA]
    if left term i same as middle term j, get ii
    if left term i different from middle term j, get ij+ji or ji+ij (terms always ordered O,A,B?)
    """

    df['O'] = 1. - df.A - df.B
    B3_middle = B3_string[1]

    B3_left2 = B2_convert(B3_string[0], B3_string[1])
    B3_right2 = B2_convert(B3_string[1], B3_string[2])
    B3 = B3_convert(B3_string)

    df["OA+AO"] = df["OA+AO"]/2.
    df["OB+BO"] = df["OB+BO"]/2.
    df["AB+BA"] = df["AB+BA"]/2.

    df["B3_Estimate"] = df[B3_left2]*df[B3_right2]/(df[B3_middle])
    df["B3sub_Estimate"] = df[B3_left2]*df[B3_string[2]] + df[B3_right2]*df[B3_string[0]] - df[B3_string[0]]*df[B3_string[1]]*df[B3_string[2]]
    return df
