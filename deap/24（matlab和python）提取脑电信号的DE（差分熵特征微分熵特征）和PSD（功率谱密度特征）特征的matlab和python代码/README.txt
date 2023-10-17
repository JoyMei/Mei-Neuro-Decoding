#信号特征提取（DE和PSD）



用于提取信号的DE（差分熵）和PSD（功率谱密度）特征的代码。



提供了Matlab和python版本。



输入：数据[n*m]n个电极，m个时间点

stft_para。频域采样率

stft_para。F每个频带的起始频率

stft_para。每个频带的防护端频率

stft_para。每个采样点的窗口长度（秒）

stft_para。fs原始频率

输出：psd，DE[n*l*k]n个电极，l个窗口，k个频带