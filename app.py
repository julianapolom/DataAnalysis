# GraphicImport
from os import stat
import sys

import matplotlib
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox

# define random variables
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
### imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from jedi.api.refactoring import inline
#%matplotlib inline
from pip._internal.utils.misc import tabulate
from scipy import stats
from tabulate import tabulate

class ejemplo_GUI(QMainWindow):

    def __init__(self):
        super().__init__()
        uic.loadUi("GUI.ui", self)
        self.cargarArchivo.clicked.connect(self.fn_cargarArchivo)
        self.cerrarAplicativo.clicked.connect(self.fn_closeEvent)
        self.procesarArchivo.clicked.connect(self.fn_procesarArchivo)
        self.comboBoxColumnas.setEditable(True)
        self.comboBoxColumnas.lineEdit().setReadOnly(True)
        #self.procesarArchivo.clicked.connect(self.Ttest_1samp)

    def fn_cargarArchivo(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'C:',
                                            'Excel (*.csv)')
        self.textArchivo.setText(fname[0])

        print(fname[0])

        dataInFile = pd.read_csv(fname[0])
        global  preprocessed_data
        preprocessed_data = dataInFile.copy()
        print(preprocessed_data);

        columnas = dataInFile.columns();
        datalist = None;
        for i, text in enumerate(columnas):
            try:
                data = datalist[i]
            except (TypeError, IndexError):
                data = None
            self.comboBoxColumnas.addItem(text, data)

    def fn_procesarArchivo(self):
        def compute_correlations(data, col):
            pearson_reg = stats.pearsonr(data[col], data["registered"])[0]
            pearson_cas = stats.pearsonr(data[col], data["casual"])[0]
            spearman_reg = stats.spearmanr(data[col], data["registered"])[0]
            spearman_cas = stats.spearmanr(data[col], data["casual"])[0]

            return pd.Series({"Pearson (registered)": pearson_reg,
                              "Spearman (registered)": spearman_reg,
                              "Pearson (casual)": pearson_cas,
                              "Spearman (casual)": spearman_cas})

        # compute correlation measures between different features
        cols = ["temp", "hum", "season", "mnth"]  # , "hum", "windspeed"]
        corr_data = pd.DataFrame(
            index=["Pearson (registered)", "Spearman (registered)", "Pearson (casual)", "Spearman (casual)"])

        for col in cols:
            print(f'Data: {col}');
            corr_data[col] = compute_correlations(preprocessed_data, col)

        #for i in corr_data.T.index:
        #    resultado = (corr_data.T["temp"][i])

        print(corr_data.T.describe())

        print(tabulate(corr_data.T, headers='keys', tablefmt='psql'))
        self.resultadoCorrelacion.setText(tabulate(corr_data.T, headers='keys', tablefmt='psql'))
        #self.T1 = corr_data.T

### Metodo de ONE SAMPLE

    def Ttest_1samp(self):
        seasons_mapping = {1: 'winter', 2: 'spring', 3: 'summer', 4: 'fall'}
        preprocessed_data['season'] = preprocessed_data['season'].apply(lambda x: seasons_mapping[x])

        # transform yr
        yr_mapping = {0: 2011, 1: 2012}
        preprocessed_data['yr'] = preprocessed_data['yr'].apply(lambda x: yr_mapping[x])

        # transform weekday
        weekday_mapping = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
        preprocessed_data['weekday'] = preprocessed_data['weekday'].apply(lambda x: weekday_mapping[x])

        # transform weathersit
        weather_mapping = {1: 'clear', 2: 'cloudy', 3: 'light_rain_snow', 4: 'heavy_rain_snow'}
        preprocessed_data['weathersit'] = preprocessed_data['weathersit'].apply(lambda x: weather_mapping[x]) 

        # transorm hum and windspeed
        preprocessed_data['hum'] = preprocessed_data['hum']*100
        preprocessed_data['windspeed'] = preprocessed_data['windspeed']*67

        # visualize preprocessed columns
        cols = ['season', 'yr', 'weekday', 'weathersit', 'hum', 'windspeed']
        print(preprocessed_data[cols].sample(10, random_state=123))
        
        population_mean = preprocessed_data.registered.mean()
        # get sample of the data (summer 2011)
        sample = preprocessed_data[(preprocessed_data.season == "summer") &(preprocessed_data.yr == 2011)].registered
        test_res = stats.ttest_1samp(sample, population_mean)
        print(f"Statistic value: {test_res[0]:.03f}, \ p-value: {test_res[1]:.03f}")

        random.seed(111)
        sample_unbiased = preprocessed_data.registered.sample(frac=0.05)
        test_result_unbiased = stats.ttest_1samp(sample_unbiased, population_mean)
        print(f"Unbiased test statistic: {test_result_unbiased[0]:.03f}, p-value: {test_result_unbiased[1]:.03f}")

    def fn_generarPairedTTest(self, event):
        columnas = self.textEditColumnas.toPlainText()
        arrayColumnas = columnas.split(",")

        res = []
        for i in range(self.comboBoxColumnas.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                res.append(self.model().item(i).data())

        resultado = stats.ttest_ind(res[0], res[1], equal_var=True)
        self.resultadoCorrelacion.setText('Resultado Paired T-Test:' + str(resultado))
    def fn_closeEvent(self, event):
        """Generate 'question' dialog on clicking 'X' button in title bar.

        Reimplement the closeEvent() event handler to include a 'Question'
        dialog with options on how to proceed - Save, Close, Cancel buttons
        """

        reply = QMessageBox.question(self, "Cerrar aplicativo", "¿Desea cerrar el aplicativo?", QMessageBox.Yes,
                                            QMessageBox.No, )

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def keyPressEvent(self, event):
        """Close application from escape key.

        results in QMessageBox dialog from closeEvent, good but how/why?
        """
        if event.key() == Qt.Key_Escape:
            self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    GUI = ejemplo_GUI()
    GUI.show()
    sys.exit(app.exec_())