import numpy as np
import matplotlib.pyplot as plt


def perform_fft(df, columns_to_fft):
    for column in columns_to_fft:
        signal = df[column]
        fft_result = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal))

        # Plot FFT results
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies[:len(frequencies) // 2], np.abs(fft_result)[:len(fft_result) // 2])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title(f'FFT of {column}')
        plt.grid(True)
        plt.show()
