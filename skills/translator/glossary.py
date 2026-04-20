"""
专有名词和缩写词词典
用于保护技术术语不被翻译
"""

# 通信领域缩写词
COMMUNICATION_ABBRS = {
    'DSSS': 'DSSS',  # Direct Sequence Spread Spectrum
    'FHSS': 'FHSS',  # Frequency-Hopping Spread Spectrum
    'CDMA': 'CDMA',  # Code Division Multiple Access
    'OFDM': 'OFDM',  # Orthogonal Frequency Division Multiplexing
    'MIMO': 'MIMO',  # Multiple-Input Multiple-Output
    'SNR': 'SNR',    # Signal-to-Noise Ratio
    'BER': 'BER',    # Bit Error Rate
    'FER': 'FER',    # Frame Error Rate
    'SER': 'SER',    # Symbol Error Rate
    'PDF': 'PDF',    # Probability Density Function
    'CDF': 'CDF',    # Cumulative Distribution Function
    'AWGN': 'AWGN',  # Additive White Gaussian Noise
    'LOS': 'LOS',    # Line of Sight
    'NLOS': 'NLOS',  # Non-Line of Sight
    'SINR': 'SINR',  # Signal-to-Interference-plus-Noise Ratio
    'RSRP': 'RSRP',  # Reference Signal Received Power
    'RSSI': 'RSSI',  # Received Signal Strength Indicator
}

# 信号处理缩写词
DSP_ABBRS = {
    'FFT': 'FFT',    # Fast Fourier Transform
    'IFFT': 'IFFT',  # Inverse Fast Fourier Transform
    'DCT': 'DCT',    # Discrete Cosine Transform
    'DWT': 'DWT',    # Discrete Wavelet Transform
    'FIR': 'FIR',    # Finite Impulse Response
    'IIR': 'IIR',    # Infinite Impulse Response
    'LMS': 'LMS',    # Least Mean Squares
    'RLS': 'RLS',    # Recursive Least Squares
    'Kalman': 'Kalman',  # Kalman filter
    'Wiener': 'Wiener',  # Wiener filter
}

# 机器学习缩写词
ML_ABBRS = {
    'CNN': 'CNN',    # Convolutional Neural Network
    'RNN': 'RNN',    # Recurrent Neural Network
    'LSTM': 'LSTM',  # Long Short-Term Memory
    'GAN': 'GAN',    # Generative Adversarial Network
    'SVM': 'SVM',    # Support Vector Machine
    'KNN': 'KNN',    # K-Nearest Neighbors
    'PCA': 'PCA',    # Principal Component Analysis
    'EM': 'EM',      # Expectation-Maximization
    'MAP': 'MAP',    # Maximum A Posteriori
    'ML': 'ML',      # Maximum Likelihood
    'MLE': 'MLE',    # Maximum Likelihood Estimation
    'MAP': 'MAP',    # Maximum A Posteriori
}

# 雷达技术缩写词
RADAR_ABBRS = {
    '雷达': 'radar',
    'SAR': 'SAR',    # Synthetic Aperture Radar
    'ISAR': 'ISAR',  # Inverse Synthetic Aperture Radar
    'MTI': 'MTI',    # Moving Target Indication
    'MTD': 'MTD',    # Moving Target Detection
    'PD': 'PD',      # Pulse Doppler
    'CFAR': 'CFAR',  # Constant False Alarm Rate
    'ROC': 'ROC',    # Receiver Operating Characteristic
    'Pd': 'Pd',      # Probability of Detection
    'Pfa': 'Pfa',    # Probability of False Alarm
    'SCR': 'SCR',    # Signal-to-Clutter Ratio
    'CNR': 'CNR',    # Clutter-to-Noise Ratio
}

# 合并所有缩写词
ALL_ABBRS = {}
ALL_ABBRS.update(COMMUNICATION_ABBRS)
ALL_ABBRS.update(DSP_ABBRS)
ALL_ABBRS.update(ML_ABBRS)
ALL_ABBRS.update(RADAR_ABBRS)


def get_abbreviation_mapping():
    """获取缩写词映射"""
    return ALL_ABBRS


def protect_abbreviations(text: str) -> tuple:
    """
    保护文本中的缩写词

    Args:
        text: 原始文本

    Returns:
        (保护后的文本, 占位符映射)
    """
    placeholders = {}
    protected_text = text
    counter = 0

    # 按长度排序，优先匹配长的缩写词
    sorted_abbrs = sorted(ALL_ABBRS.keys(), key=len, reverse=True)

    for abbr in sorted_abbrs:
        # 使用单词边界匹配
        import re
        pattern = r'\b' + re.escape(abbr) + r'\b'

        def replace_func(match):
            nonlocal counter
            placeholder = f"ABBRPLACEHOLDER{counter}ABBR"
            placeholders[placeholder] = abbr
            counter += 1
            return placeholder

        protected_text = re.sub(pattern, replace_func, protected_text)

    return protected_text, placeholders


def restore_abbreviations(text: str, placeholders: dict) -> str:
    """
    恢复缩写词

    Args:
        text: 包含占位符的文本
        placeholders: 占位符映射

    Returns:
        恢复后的文本
    """
    for placeholder, abbr in placeholders.items():
        text = text.replace(placeholder, abbr)

    return text
