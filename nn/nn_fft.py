import numpy as np
import matplotlib.pyplot as plt

def perform_fft(df, columns_to_fft):#columns_to_fft list of column names
    for column in columns_to_fft:#This loop iterates over each column specified in the columns_to_fft list.
        signal = df[column]
        fft_result = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal))

        # Plot FFT results
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies[:len(frequencies) // 2], np.abs(fft_result)[:len(fft_result) // 2])
        plt.xlabel('Frequency (Hz)')#on the x-axis
        plt.ylabel('Amplitude')
        plt.title(f'FFT of {column}')
        plt.grid(True)
        plt.show()
