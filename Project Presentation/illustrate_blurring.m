function [ ] = illustrate_blurring( )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

x = 0:0.1:12;
y = gaussmf(x,[2 7]);
y2 = gaussmf(x,[1.5 5]);
plot(x,y+.8*y2);

hold all;
plot(x,y2+.8*y);
xlabel('time step');
ylabel('fluorescence output');
title('Blurred Fluorescence Output');
end

