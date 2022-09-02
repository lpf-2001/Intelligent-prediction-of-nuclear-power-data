import xlwt

def Export_Data(a_Test, a_predict, col):

    book = xlwt.Workbook(encoding='utf-8', style_compression=0) #创建文件
    sheet = book.add_sheet('Sheet1', cell_overwrite_ok=True) #创建表格
    book_name = ('入口温度', '出口温度', '破损SG蒸汽温度') #自定义文件名


    col_name = ('入口温度原始值', '入口温度预测值', '出口温度原始值', '出口温度预测值', '破损SG蒸汽温度原始值', '破损SG蒸汽温度预测值') #自定义列名
    sheet.write(0, 0, col_name[col * 2]) #将列名写入表单
    sheet.write(0, 1, col_name[col * 2 + 1]) #将列名写入表单


    #将数据写入表单
    for i in range(len(a_Test)):
        sheet.write(i+1, 0, str(a_Test[i]))
        sheet.write(i+1, 1, str(a_predict[i][0]))

    #保存文件
    savepath = r'C:\Users\86178\Desktop\项目组\new\预测数据_LSTM_{}.xls'.format(book_name[col])
    book.save(savepath)