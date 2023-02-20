import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QColor
import pyqtgraph as pg
import wave
import scipy
import pyaudio
import time
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.left = 50
        self.top = 50
        self.title = 'Digital Signal Processing App'
        self.width = 800
        self.height = 900

        self.penRed = pg.mkPen(color=QColor(255, 0, 0))  #red
        self.penBlue = pg.mkPen(color=QColor(0, 0, 255))  #blue

        self.CHUNK = 2048


        self.file = wave.open('MonoSample.wav')
        self.data = self.file.readframes(self.file.getnframes())
        self.inputData = np.frombuffer(self.data, dtype=np.int16)

        self.s_rate = self.file.getframerate()
        self.nyquist = 0.5 * self.s_rate
        self.nFrames = self.file.getnframes()
        self.duration = self.nFrames/self.s_rate

        print("Die Audiodatei hat eine Länge von :", self.duration, " Sekunden")
        self.p = pyaudio.PyAudio()

        #PyAudio Stream öffnen
        self.stream = self.p.open(format=self.p.get_format_from_width(self.file.getsampwidth()),
                        channels=self.file.getnchannels(),
                        rate=self.file.getframerate(),
                        output=True)

        self.time = time.time()
        self.audioData = self.processData()
        print("Das Signal wurde innerhalb von: ",time.time()-self.time, " Sekunden verarbeitet")
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        #Processed Audio Waveform
        self.plot1 = pg.PlotWidget()
        self.plot1.setYRange(self.audioData.max()* -1,self.audioData.max())
        self.plot1.setBackground(QColor(255, 255, 255))
        layout.addWidget(self.plot1)

        # Processed Frequencyspektrum plot
        self.plot2 = pg.PlotWidget()
        self.plot2.setLogMode(x=True, y=False)
        self.plot2.setBackground(QColor(255, 255, 255))
        layout.addWidget(self.plot2)

        #Original Audio Waveform
        self.plot3 = pg.PlotWidget()
        self.plot3.setBackground(QColor(255,255,255))
        self.plot3.setYRange(self.inputData.max()* -1,self.inputData.max())
        layout.addWidget(self.plot3)

        #Original Frequenzspektrum plot
        self.plot4 = pg.PlotWidget()
        self.plot4.setLogMode(x=True, y=False)
        self.plot4.setBackground(QColor(255, 255, 255))
        layout.addWidget(self.plot4)

        self.x = [i for i in range(2048)]
        self.y1 = [0 for i in range(2048)]
        self.y2 = np.zeros(1024)
        self.y3 = [0 for i in range(2048)]
        self.y4 = np.zeros(1024)

        self.show()
        self.update_plots()
        self.p.close()
        print("Visualisierung der Audiosinale ist Fertiggestellt")

    def update_plots(self):
        for i in range(int(self.file.getnframes()/self.CHUNK)):
            self.y1 = self.audioData[i*self.CHUNK:(i+1)*self.CHUNK]
            #self.time = time.time()
            self.y2 = np.fft.rfft(self.y1)
            #print(time.time()-self.time)
            self.y2 = np.abs(self.y2)

            self.y3 = self.inputData[i*self.CHUNK:(i+1)*self.CHUNK]
            self.y4 = np.fft.rfft(self.y3)
            self.y4 = np.abs(self.y4)

            self.plot1.clear()
            self.plot1.plot(self.x, self.y1, pen=self.penBlue)

            self.plot2.clear()
            self.plot2.plot((np.arange(len(self.y2)) * (20000/1024))/2, self.y2, pen=self.penBlue)

            self.plot3.clear()
            self.plot3.plot(self.x, self.y3, pen=self.penRed)

            self.plot4.clear()
            self.plot4.plot((np.arange(len(self.y4)) * (20000/1024))/2, self.y4, pen=self.penRed)

            self.stream.write(self.y1.astype(np.int16).tobytes())
            QApplication.processEvents()

    def processData(self):
        self.audioData = self.filter_LpHp(self.inputData, 19000,21, 5)
        self.audioData = self.delay(self.audioData)
        return self.audioData


    def delay(self,input):
        delay = np.zeros(input.size)
        delay[0] = 1
        delay[int(0.1 * self.s_rate)] = 0.5
        delay[int(0.4 * self.s_rate)] = 0.2
        delay_return  = scipy.signal.convolve(input, delay)
        return delay_return[0:input.size]

    def filter_LpHp(self,input, cutoff_lp, cutoff_hp, order):
        lp_cutoff = cutoff_lp / self.nyquist
        hp_cutoff = cutoff_hp / self.nyquist
        # lowpassfilter
        b, a = scipy.signal.butter(order, lp_cutoff, btype='low')
        y = scipy.signal.filtfilt(b, a, input)
        # highpassfilter
        b, a = scipy.signal.butter(order, hp_cutoff, btype='high')
        y_return = scipy.signal.filtfilt(b, a, y)
        return y_return


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())