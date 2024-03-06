

import numpy as np
import matplotlib.pyplot as plt

def perform_fft(df, columns_to_fft):#column_to_fft list of column names
    for column in columns_to_fft:# This loop iterates over each column specified in the columns_to_fft list.
        signal = df[column]
        fft_result = np.fft.fft(signal)#apply the algorithm on the signals
        frequencies = np.fft.fftfreq(len(signal))#it calculates the frequencies.

        # Plot FFT results
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies[:len(frequencies) // 2], np.abs(fft_result)[:len(fft_result) // 2])
        #only using the positive values not using the negative values.
        #np.abs calculates the strenth of each frequency component in the signal.
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title(f'FFT of {column}')
        plt.grid(True)
        plt.show()
