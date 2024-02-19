import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import inquirer
import time

use_percentages = 1
reference = ['ch', 'm', 'ml'] #reference for percentages

xabbreviation_type = "Bytes"
yabbreviation_type = "Normal"


def abbreviate_number(number, abbreviation_type = "Normal"):
    #convert number from string to float
    number = number.replace('\u2212', '-')
    number = float(number)
    if number >= 1000:
        if abbreviation_type == "Bytes":
            #print(number)
            abbreviations = ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei"]
            base = 1024
        elif abbreviation_type == "Normal":
            abbreviations = ["", "K", "M", "B"]
            base = 1000
        else:
            return "Invalid abbreviation type."
        magnitude = 0
        while abs(number) >= base and magnitude < len(abbreviations) - 1:
            number /= base
            magnitude += 1
        #print(number)
        if number % 1 != 0: number = round(number, 1)
        if number % 1 == 0: number = int(number)
        return f"{number}{abbreviations[magnitude]}"
    
    if number % 1 == 0: number = int(number)
    return number

bnw =['lightgray','dimgray','darkgray','black','silver','gray','darkslategray','slategray','lightslategrey','gainsboro','whitesmoke','white']

files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith(".csv")]

max_length = max(len(f) for f in files) if files else 0

choices = [f"{f.ljust(max_length)} - Created on: {time.ctime(os.path.getctime(f))}" for f in files]

if choices:
    questions = [
        inquirer.List('file',
                      message="Select a file",
                      choices=choices,
                     ),
    ]

    answers = inquirer.prompt(questions)

    selected_file = answers['file'].split(' - ')[0].strip()
    print(f"You selected: {selected_file}")
else:
    print("No CSV files found in the current directory.")

df = pd.read_csv(selected_file, skipinitialspace=True, engine='python', na_values=[''])
grouped = df.groupby(['operation', 'x', 'z', 'xtype', 'ytype', 'xunit', 'yunit'], dropna=False)[['y']].mean().reset_index()
operations = grouped['operation'].unique()

for operation in operations:
    df = grouped[(grouped['operation'] == operation)]

    for xtype, xunit in df[['xtype', 'xunit']].drop_duplicates().values.tolist():

        for ytype, yunit in df[['ytype', 'yunit']].drop_duplicates().values.tolist():

            df_i = df[df['ytype'] == ytype]
            df_i = df_i.pivot(index='x', columns='z', values='y')
            df_i = df_i.apply(pd.to_numeric, errors='coerce')
            print(df_i)

            if use_percentages:
                for ref in reference:
                    if ref in df_i.columns:
                        df_m = df_i[ref].copy()
                        for column in df_i.columns:
                            df_i[column] = df_m.sub(df_i[column]).div(df_m).mul(100)
                        break
                #remove the reference column
                df_i = df_i.drop(columns=ref)

            #fig, axes = plt.subplots(nrows=2, gridspec_kw=dict(height_ratios=[3,1]))
            fig, axes = plt.subplots(nrows=1)
            df_i.plot(ax = axes, kind='bar', use_index = True, width=.8, color=bnw, edgecolor='black', linewidth=.1)
            axes.legend(loc = "best", fontsize="5", ncol=3)
            
            axes.tick_params(axis='both', which='major', labelsize=6)
            ylabel = ytype if not use_percentages else f"% of execution \n{ytype} reduction"

            if use_percentages:
                ylabel = f"% of reduction\nin {ytype.lower()} "
            else:
                ylabel = f"{ytype} ({yunit})" if yunit != "None" else f"{ytype}"

            axes.set_ylabel(ylabel, fontsize=6)
            axes.set_xlabel(xtype, fontsize=6)
            
            
            x_labels = [abbreviate_number(x._text, xabbreviation_type) for x in axes.get_xticklabels()]
            #print(x_labels)
            axes.set_xticklabels(x_labels)

            ylabels = [i for i in axes.get_yticklabels()]
            axes.set_yticks(axes.get_yticks())
            y_labels = [abbreviate_number(x._text, yabbreviation_type) for x in axes.get_yticklabels()]
            #axes.yaxis.set_major_locator(mticker.FixedLocator(y_labels))
            #get ticks
            axes.set_yticklabels(y_labels)


            #table
            """
            rowslabels = df.columns.tolist()
            colslabels = ['min', 'mean', 'max']
            tabledata = []
            for i in rowslabels:
                data = [x for x in [df[i].min(), df[i].mean(), df[i].max()]]
                try:
                    #abbreviate numbers according to the operation
                    data = [abbreviate_number(x, "Bytes" if "memory" in operation else "Normal") for x in data]    
                except BaseException:
                    pass
                tabledata.append(data)
            axes[1].axis('off')
            axes[1] = plt.table(
                cellText=tabledata,
                rowLabels=rowslabels,
                colLabels=colslabels,
                loc='center',
                cellLoc='center',
                colWidths=[.12 for x in colslabels])
            axes[1].scale(1,1.7)    
            axes[1].auto_set_font_size(False)
            axes[1].set_fontsize(5)
            """


            fig.set_size_inches(2, 1.33)
            #fig.set_size_inches(2, 1.2)
            fig.tight_layout(h_pad=0, w_pad=0)

            #make the figure as compact as possible
            fig.subplots_adjust(top=.95, bottom=.35, left=.25, right=.99)

            #save figure in svg format in folder "plots" and create folder if it doesn't exist
            
            #plt.show()
            #plt.show()
            folder_name = f"plots"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            
            file_name = f"{operation}_{xtype}_{ytype}"
            file_name = file_name.replace(" ", "")

            path = os.path.join(folder_name, file_name)

            plt.savefig(path + ".pdf", format="pdf")
            df_i.to_csv(path + ".csv")
