import pandas as pd

xls = pd.ExcelFile('Engenharia Clínica - Atividade 2.xlsx')
df1 = pd.read_excel(xls, 'Centro Cirúrgico')
df2 = pd.read_excel(xls, 'Exames Diagnósticos')