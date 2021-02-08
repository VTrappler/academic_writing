# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

exec(open('/home/victor/acadwriting/Manuscrit/plots_settings.py').read())



def read_csv_sobolrep(filenames, variable_name, order = ['1', '2', 'T']):
    df_tot = []
    for filen, order_ in zip(filenames, order):
        df = pd.read_csv(filen)
        print(df)
        df = df.rename(columns = {'Unnamed: 0': 'Variable'})
        try:
            df['Variable'] = variable_name
        except ValueError:
            pass
        df['Sobol'] = df['original'] - df['bias']
        df['CI'] = df['max. c.i.'] - df['min. c.i.']
        df['order'] = order_
        df_tot.append(df)
    return pd.concat(df_tot)




# minCI = df['min. c.i.'].values
# maxCI = df['max. c.i.'].values
# yerr = np.vstack([minCI, maxCI])
# df['1orT'] = df['order'].isin(['1', 'T'])
# sns.barplot(x='Variable', y='Sobol', data=df, hue='order', yerr=df[['CI', 'CI']].values.T)
# plt.show()


def make_plot_sobol(df, variable_name, figname=None, dollar=False):
    x = np.arange(len(df[df['order'] == '1']['Variable']))
    width = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=col_full)

    rects1 = ax1.barh(x - width / 2, df[df['order'] == '1']['Sobol'], width, label='First order',
                      xerr=df[df['order'] == '1']['CI'])
    rectsT = ax1.barh(x + width / 2, df[df['order'] == 'T']['Sobol'], width, label='Total order',
                      xerr=df[df['order'] == 'T']['CI'])
    ax1.set_xlabel("Sobol indices")
    ax1.set_title('First and Total order indices')
    ax1.set_yticks(x)
    ax1.set_yticklabels(df['Variable'])
    ax1.legend(loc='lower right')
    ax1.set_xlim(left=0)
    ax1.invert_yaxis()
    x2 = np.arange( int(len(x) * (len(x) - 1) / 2))

    rects2 = ax2.barh(x2, df[df['order'] == '2']['Sobol'], label='Second order',
                      xerr=df[df['order'] == '2']['CI'], color=colors[2])
    ax2.set_xlim([0, 1])
    inde = [(int(xij[1]) - 1, int(xij[2]) - 1)
            for xij in df[df['order'] == '2']['Variable'].values]
    if dollar:
        xt = ['${}\\times {}$'.format(variable_name[i][1:-1], variable_name[j][1:-1]) for i, j in inde]
    else:
        xt = ['{} $\\times$ {}'.format(variable_name[i], variable_name[j]) for i, j in inde]

    ax2.set_title(r'Second order indices')
    ax2.set_yticks(x2)
    ax2.set_yticklabels(xt)
    ax2.tick_params(axis='y', which='major', labelsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Sobol indices")
    ax2.legend(loc='lower right')
    plt.tight_layout()
    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)
    plt.close()


if __name__ == '__main__':
    tides_names = [r'$M_2$', r'$S_2$', r'$N_2$', r'$K_2$', r'$O_1$']
    filenames = ['/home/victor/acadwriting/Manuscrit/Text/Chapter5/tides_{}.csv'.format(filen)
             for filen in ['S', 'S2', 'T']]
    df_tides = read_csv_sobolrep(filenames, variable_name=tides_names)
    make_plot_sobol(df_tides, variable_name=tides_names, dollar=True,
                    figname='/home/victor/acadwriting/Manuscrit/Text/Chapter5/img/SA_tides.pgf')

    sed_names = ['R', 'C', 'G', 'S', 'SF', 'Si,V']
    filenames = ['/home/victor/acadwriting/Manuscrit/Text/Chapter5/sed_{}.csv'.format(filen)
                 for filen in ['S', 'S2', 'T']]
    df_sed = read_csv_sobolrep(filenames, variable_name=sed_names)
    make_plot_sobol(df_sed, variable_name=sed_names,
                    figname='/home/victor/acadwriting/Manuscrit/Text/Chapter5/img/SA_sediments.pgf')


# EOF ----------------------------------------------------------------------
