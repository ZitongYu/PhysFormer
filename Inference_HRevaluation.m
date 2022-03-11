clear
clc

load Inference_Physformer_TDC07_sharp2_hid96_head4_layer12_VIPL.mat 

GT_list = importdata('VIPL_fold1_test1.txt');

total_samples = length(outputs_rPPG_concat);

HR_peaks = [];
HR_PSD = [];
HR_GT = [];

signal =  double(outputs_rPPG_concat);

framerate = GT_list.data(1,2);
GT_HR = GT_list.data(1,3);
   
   
%    signal_filtered = signal;
signal_filtered = bpfilter64(signal, framerate);
signal_filtered = (signal_filtered-mean(signal_filtered))/std(signal_filtered);


   
%% PSD for HR 
%     % Single long clip
%    [Pg,f] = pwelch(signal_filtered,[],[],2^13,framerate);
%     Frange = find(f>0.7&f<3); % consider the frequency within [0.7Hz, 4Hz].
%     idxG = Pg == max(Pg(Frange));
%     HR2 = f(idxG)*60;

%     % Separate into three clips
signal_length = length(signal_filtered);
[Pg,f] = pwelch(signal_filtered(1:floor(signal_length/3)),[],[],2^13,framerate);
Frange = find(f>0.7&f<4); % consider the frequency within [0.7Hz, 4Hz].
idxG = Pg == max(Pg(Frange));
HR2_1 = f(idxG)*60;
[Pg,f] = pwelch(signal_filtered(floor(signal_length/3):2*floor(signal_length/3)),[],[],2^13,framerate);
Frange = find(f>0.7&f<4); % consider the frequency within [0.7Hz, 4Hz].
idxG = Pg == max(Pg(Frange));
HR2_2 = f(idxG)*60;
[Pg,f] = pwelch(signal_filtered(2*floor(signal_length/3):signal_length),[],[],2^13,framerate);
Frange = find(f>0.7&f<4); % consider the frequency within [0.7Hz, 4Hz].
idxG = Pg == max(Pg(Frange));
HR2_3 = f(idxG)*60;
HR2 = (HR2_1+HR2_2+HR2_3)/3;

HR_peaks = [HR_peaks; HR1];
HR_PSD = [HR_PSD; HR2];
HR_GT = [HR_GT; GT_HR];




%% calculate ErrorMean, ErrorSD, RMSE, R

Error_PSD = HR_PSD - HR_GT;
MAE = abs(Error_PSD)



    



