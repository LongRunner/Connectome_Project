function [ output_args ] = illustrate_timestep (  )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
x = 0:1:100;
y = gaussmf(x,[2 50]);
y2 = gampdf(x,1.5, 51);
 y3 = gampdf(x,1.5,51);
  y4 = gampdf(x,1.5,51);
  idx=1;
  while(idx<=101)
      if(idx<=50)
          y3(idx)=rand/2000;
          y4(idx)=rand/2000;
      elseif(idx>50 && idx<=55)
          y3(idx)=y2(idx)+rand/2000;
          y4(idx)=y2(idx)+rand/2000;
      elseif(idx>55 && idx<65)
          y3(idx)=y3(idx-1)/1.5+rand/2000;
          y4(idx)=y4(idx-1)/1.5+rand/2000;
      else
          y3(idx)=rand/2000;
          y4(idx)=rand/2000;
      end
      
          
      idx = idx+1;
  end
  
scatter(x,y3);
hold all;
scatter(x,y4);
xlabel('true time');
ylabel('fluorescence output');
title('True Fluorescence Activity');


end

