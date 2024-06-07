import pandas as pd 
from openpyxl import Workbook, load_workbook
import pdb
from os import scandir, getcwd
import os
from datetime import datetime
import re
import openpyxl

# input_file = input("Ingrese nombre del archivo xlsx: ")
# DIR = input_file + ".xlsx"

def ls(ruta = getcwd()):
    return [arch.name for arch in scandir(ruta) if arch.is_file()]

archivos =  ls("recibidos")
try:
    for arch in archivos:
        DIR = "recibidos/"+arch
        archivo = pd.read_excel(DIR, header=1)
        if "csv" in arch:
            archivo = pd.read_csv(DIR)
        print(DIR)
        total_sum = 0
        wb = load_workbook(filename = DIR)
        ws1 = wb.active

        sheet_ranges = wb['Sheet1']
        #pdb.set_trace()
        try:
            fecha = datetime.strptime(ws1['A3'].value, '%d/%m/%Y').strftime('%m/%Y') 
        except: 
            print('Error en la fecha')
            print(ws1['A3'].value)  
            print(arch)
        ws1['S17'] = "Factura A" 
        ws1['T17'] = "Nota de Crédito B" 
        ws1['U17'] = "Factura B" 
        ws1['V17'] = "Factura C"
        ws1['W17'] = "Nota de Crédito A"
        ws1['X17'] = "Factura de Crédito Electrónica MyPyMEs A"
        ws1['Y17'] = "12 - Nota de Débito C"

        ws1['R18'] = "Imp. Neto Gravado" 
        ws1['R19'] = "Imp. Neto No Gravado" 
        ws1['R20'] = "IVA" 
        ws1['R21'] = "TOTAL" 

        # NETO GRAVADO
        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Neto Gravado"][data]
            if Tipo == "1 - Factura A":
                total_sum = total_sum + valor_data

        total_sum = "{0:.2f}".format(total_sum)

        ws1['S18'] = total_sum
        #print(total_sum)

        total_sum = 0
        
        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Neto Gravado"][data]
            if Tipo == "8 - Nota de Crédito B":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['T18'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Neto Gravado"][data]
            if Tipo == "6 - Factura B":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['U18'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Neto Gravado"][data]
            if Tipo == "11 - Factura C":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['V18'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Neto Gravado"][data]
            if Tipo == "3 - Nota de Crédito A":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['W18'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Neto Gravado"][data]
            if Tipo == "201 - Factura de Crédito Electrónica MyPyMEs (FCE) A":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['X18'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Neto Gravado"][data]
            if Tipo == "12 - Nota de Débito C":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['Y18'] = total_sum
        #print(total_sum)

        total_sum = 0
        # NETO NO GRAVADO
        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Neto No Gravado"][data]
            if Tipo == "1 - Factura A":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['S19'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Neto No Gravado"][data]
            
            if Tipo == "8 - Nota de Crédito B":
                total_sum = total_sum + valor_data

        total_sum = "{0:.2f}".format(total_sum)
        ws1['T19'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Neto No Gravado"][data]
            if Tipo == "6 - Factura B":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['U19'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Neto No Gravado"][data]
            if Tipo == "11 - Factura C":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['V19'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Neto No Gravado"][data]
            if Tipo == "3 - Nota de Crédito A":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['W19'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Neto No Gravado"][data]
            if Tipo == "201 - Factura de Crédito Electrónica MyPyMEs (FCE) A":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['X19'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Neto No Gravado"][data]
            if Tipo == "12 - Nota de Débito C":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['Y19'] = total_sum
        #print(total_sum)

        total_sum = 0

        # IVA

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["IVA"][data]
            
            if Tipo == "1 - Factura A":
                total_sum = total_sum + valor_data

        total_sum = "{0:.2f}".format(total_sum)
        ws1['S20'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["IVA"][data]
            
            if Tipo == "8 - Nota de Crédito B":
                total_sum = total_sum + valor_data

        total_sum = "{0:.2f}".format(total_sum)
        ws1['T20'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["IVA"][data]
            if Tipo == "6 - Factura B":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['U20'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["IVA"][data]
            if Tipo == "11 - Factura C":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['V20'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["IVA"][data]
            if Tipo == "3 - Nota de Crédito A":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['W20'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["IVA"][data]
            if Tipo == "201 - Factura de Crédito Electrónica MyPyMEs (FCE) A":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['X20'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["IVA"][data]
            if Tipo == "12 - Nota de Débito C":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['Y20'] = total_sum
        #print(total_sum)

        total_sum = 0

        # TOTALES
        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Total"][data]
            
            if Tipo == "1 - Factura A":
                total_sum = total_sum + valor_data

        total_sum = "{0:.2f}".format(total_sum)
        ws1['S21'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Total"][data]
            
            if Tipo == "8 - Nota de Crédito B":
                total_sum = total_sum + valor_data

        total_sum = "{0:.2f}".format(total_sum)
        ws1['T21'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Total"][data]
            if Tipo == "6 - Factura B":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['U21'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Total"][data]
            if Tipo == "11 - Factura C":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['V21'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Total"][data]
            if Tipo == "3 - Nota de Crédito A":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['W21'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Total"][data]
            if Tipo == "201 - Factura de Crédito Electrónica MyPyMEs (FCE) A":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['X21'] = total_sum
        #print(total_sum)

        total_sum = 0

        for data in archivo.index:
            Tipo = archivo["Tipo"][data]
            valor_data = archivo["Imp. Total"][data]
            if Tipo == "12 - Nota de Débito C":
                total_sum = total_sum + valor_data
        total_sum = "{0:.2f}".format(total_sum)
        ws1['Y21'] = total_sum
        #print(total_sum)

        total_sum = 0

        # VALOR IVA
        valor = 0
        for data in archivo.index:
            imp_total = archivo["Imp. Total"][data]
            neto = archivo["Imp. Neto Gravado"][data]
            if imp_total != 0 and neto != 0:
                valor = imp_total / neto
            else:
                valor = neto
                #print(valor)
                #print(neto)
                
            total_sum = "{0:.2f}".format(valor)

        wb.save(filename = DIR)
        
        clientes_excel = pd.read_excel("clientes.xls", header=0, sheet_name="VERO2023")
        pa = pd.DataFrame(clientes_excel)
        cuil = [int(cuil) for cuil in re.findall(r'-?\d+', arch)]
        cliente_nombre = pa[pa['USUARIOS'] == cuil[0]]['CLIENTES'].to_string(index=False)

    
        if 'Mis Comprobantes Emitidos' in arch:
            tipo_libro='Libro venta '
            #pdb.set_trace()
        else:
            tipo_libro='Libro compra '
            #pdb.set_trace()
            
        date = fecha.replace('/', '-') 
        final = tipo_libro+cliente_nombre+ ' ' + date + '.xlsx'
        if cliente_nombre == 'Series([], )' or cliente_nombre == "NaN" or cliente_nombre in "Series([], )":
            final = arch
    
        #pdb.set_trace()
        DIR_final = 'terminado/'+ final
        wb.save(DIR_final)

except NameError:
    print(NameError)
    pdb.set_trace()

#date = fecha.replace('/', '-') 
#final = tipo_libro+cliente_nombre+ ' ' + date + '.xlsx'
#print (final)
#pdb.set_trace()




    #ubi = DIR.replace('recibidos/', '')
    #nombre = ubi.replace('.xlsx', ' ') + fecha + '.xlsx'
    
#print('todo correcto')

#pd.Series(cliente_nombre, dtype=pd.StringDtype)

    #os.rename(ubi, nombre)
    #ubi_nueva = os.path.join(os.getcwd(), nombre)
   # ubi_vieja = os.path.join(os.getcwd(), ubi)
   # os.rename(ubi_nueva ,ubi_vieja)

    #  os.getcwd()+'\\'+ubi
    # os.rename(ubi, nombre)
    #os.path.join(os.getcwd(), ubi)
    #os.path.join(os.getcwd(), nombre)
    # archivo = "/Users/PC/Documents/comprobantes/recibidos/" + ubi
    # nombre_nuevo = "/Users/PC/Documents/comprobantes/recibidos/archivo_nuevo.txt"
    # os.rename(archivo, nombre_nuevo)