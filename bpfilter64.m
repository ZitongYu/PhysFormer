%% 64 bandpass filter using hamming window at [0.8, 3]Hz.
function [s2f] = bpfilter64(s2,fs)
% fs = 15;
minfq = 0.8*2/fs; %0.8
maxfq = 3*2/fs;  %3
% maxfq = 3*2/fs;
fir1_len = round(length(s2)/10);
bpfilter = fir1(fir1_len,[minfq maxfq]);
% bpfilter = fir1(64,[minfq maxfq]);

s2f = filtfilt(bpfilter,1,s2);