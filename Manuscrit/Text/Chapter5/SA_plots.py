# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# df = pd.read_csv('/home/victor/acadwriting/Manuscrit/Text/Chapter5/tides_S.csv')
# df = df.rename(columns = {'Unnamed: 0': 'Variable'})
# df['Variable'] = [r'M2', r'S2', r'N2', r'K2', r'O1']
# sns.barplot(x='Variable', y='original', yerr =df['min. c.i.'] , data=df)
# CI = df['max. c.i.'] + df['min. c.i.']
# sns.pointplot(x='Variable', ci=CI, data=df)
# plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teststr', type=str)
    parser.add_argument('--testint', type=int)
    args = parser.parse_args()
    aa = [0, 1, 2, 20]
    print(aa[args.testint])
    print(type(args.teststr))
    print(args.teststr + 'hhahaha')
